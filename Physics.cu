/*
 * Physics.cu
 *
 *  Created on: Mar 10, 2023
 *      Author: bongwon
 */

#include "header/Physics.cuh"

void setConfig(
	float vDiffusion = 0.8f, // 0.8f
	float pressure = 1.2f, // 1.5f
	float vorticity = 50.0f, // 50.0f
	float cDiffuion = 0.8f, // 0.8f
	float dDiffuion = 1.2f, // 1.2f
	float force = 5000.0f, // 5000.0f
	float bloomIntense = 0.1f,
	int radius = 400, // 400
	bool bloom = true
)
{
	config.velocityDiffusion = vDiffusion;
	config.pressure = pressure;
	config.vorticity = vorticity;
	config.colorDiffusion = cDiffuion;
	config.densityDiffusion = dDiffuion;
	config.forceScale = force;
	config.bloomIntense = bloomIntense;
	config.radius = radius;
	config.bloomEnabled = bloom;
}

PhysicsContainer::PhysicsContainer() {this->xSize = 0; this->ySize=0;}
PhysicsContainer::PhysicsContainer(int p_xSize, int p_ySize) {this->xSize = p_xSize; this->ySize = p_ySize;}
PhysicsContainer::~PhysicsContainer() {}

// inits all buffers, must be called before computeField function call
void PhysicsContainer::cudaInit()
{
	setConfig();

	colorArray[0] = { 1.0f, 0.0f, 0.0f };
	colorArray[1] = { 0.0f, 1.0f, 0.0f };
	colorArray[2] = { 1.0f, 0.0f, 1.0f };
	colorArray[3] = { 1.0f, 1.0f, 0.0f };
	colorArray[4] = { 0.0f, 1.0f, 1.0f };
	colorArray[5] = { 1.0f, 0.0f, 1.0f };
	colorArray[6] = { 1.0f, 0.5f, 0.3f };

	int idx = rand() % colorArraySize;
	currentColor = colorArray[idx];

	cudaMalloc(&colorField, xSize * ySize * sizeof(float4));

	// Gauss-Seidel 방식 대신에 Jacobian 방식으로 linear solver를 진행할 것이라서
	// old와 new 버전 필요
	cudaMalloc(&oldField, xSize * ySize * sizeof(Particle)); // 속도에 대한 벡터장
	cudaMalloc(&newField, xSize * ySize * sizeof(Particle)); // 속도에 대한 벡터장

	cudaMalloc(&pressureOld, xSize * ySize * sizeof(float)); // 어디서는 간단한 수식을 위해 pressure를 constant로 두는데
	cudaMalloc(&pressureNew, xSize * ySize * sizeof(float)); // 우리는 일단, pressure도 계산되는 값으로 두자ㅣ

	// 2D에선 xy축에 수직인 z축이 항상 vorticity의 방향이므로, 방향을 가리키는 벡터를 생략하고,
	// 그 크기인 scalar만 vorticity에 해당시킬 수 있다!
	// 따라서 2D에선 vorticity는 상수로 남길 수 있다.
	cudaMalloc(&vorticityField, xSize * ySize * sizeof(float));
}

void PhysicsContainer::cudaExit()
{
	cudaFree(colorField);
	cudaFree(oldField);
	cudaFree(newField);
	cudaFree(pressureOld);
	cudaFree(pressureNew);
	cudaFree(vorticityField);
}

// computes curl of velocity field
// 아, 내가 divergence 식이랑 헷갈렸네. 이거 rot 식이라서, curlㅣ네 ㅇㅇ. 내가 바보였음.
__device__ float curl(Particle* field, size_t xSize, size_t ySize, int x, int y)
{
	Vec2 C = field[int(y) * xSize + int(x)].u;
#define SET(P, x, y) if (x < xSize && x >= 0 && y < ySize && y >= 0) P = field[int(y) * xSize + int(x)]
	float x1 = -C.x, x2 = -C.x, y1 = -C.y, y2 = -C.y;
	SET(x1, x + 1, y).u.x;
	SET(x2, x - 1, y).u.x;
	SET(y1, x, y + 1).u.y;
	SET(y2, x, y - 1).u.y;
#undef SET
	float res = ((y1 - y2) - (x1 - x2)) * 0.5f; // rot 식
	return res;
}

// computes absolute value gradient of vorticity field
__device__ Vec2 absGradient(float* field, size_t xSize, size_t ySize, int x, int y)
{
	float C = field[int(y) * xSize + int(x)];
#define SET(P, x, y) if (x < xSize && x >= 0 && y < ySize && y >= 0) P = field[int(y) * xSize + int(x)]
	float x1 = C, x2 = C, y1 = C, y2 = C;
	SET(x1, x + 1, y);
	SET(x2, x - 1, y);
	SET(y1, x, y + 1);
	SET(y2, x, y - 1);
#undef SET

	Vec2 res = { (abs(x1) - abs(x2)) * 0.5f, (abs(y1) - abs(y2)) * 0.5f };
	return res;
}

// performs iteration of jacobi method on velocity grid field
__device__ Vec2 jacobiVelocity(Particle* field, size_t xSize, size_t ySize, Vec2 v, Vec2 B, float alpha, float beta)
{
	Vec2 vU = B * -1.0f, vD = B * -1.0f, vR = B * -1.0f, vL = B * -1.0f;
#define SET(U, x, y) if (x < xSize && x >= 0 && y < ySize && y >= 0) U = field[int(y) * xSize + int(x)].u
	SET(vU, v.x, v.y - 1);
	SET(vD, v.x, v.y + 1);
	SET(vL, v.x - 1, v.y);
	SET(vR, v.x + 1, v.y);
#undef SET
	// 장봉원 : Gauss-Seidel relaxation이 아니라 Jacobian 방식으로 한다는 건가?
	// https://www.quora.com/What-is-the-difference-between-the-Gauss-Seidel-and-the-Jacobi-Method
	// 둘 다 linear solver로써 역할.
	// 다만, Jacobi는 이전 iteration 결과물을 사용
	// Gauss-Seidel은 최신 업데이트된 값을 사용
	// 따라서, Gauss-Seidel은 CUDA로 구현한다면, race condition 문제가 발생할 수 있다.
	// 물론, 어떤 강의자료에선 Gauss-Seidel이 Jacobi의 개선판이라고 하지만.. 일단, 나는 구현이 중요하므로 개선여부를 따지지 않겠다.
	// 혹시나 궁금해서 찾아본, Gauss-Seidel이 진짜로 수렴하는 예제 : https://www.youtube.com/watch?v=HqgLVk-eCno
	v = (vU + vD + vL + vR + B * alpha) * (1.0f / beta);
	return v;
}

// computes divergency of velocity field
__device__ float divergency(Particle* field, size_t xSize, size_t ySize, int x, int y)
{
	Particle& C = field[int(y) * xSize + int(x)];
	float x1 = -1 * C.u.x, x2 = -1 * C.u.x, y1 = -1 * C.u.y, y2 = -1 * C.u.y;
#define SET(P, x, y) if (x < xSize && x >= 0 && y < ySize && y >= 0) P = field[int(y) * xSize + int(x)]
	SET(x1, x + 1, y).u.x;
	SET(x2, x - 1, y).u.x;
	SET(y1, x, y + 1).u.y;
	SET(y2, x, y - 1).u.y;
#undef SET
	return (x1 - x2 + y1 - y2) * 0.5f;
}

// performs iteration of jacobi method on pressure grid field
__device__ float jacobiPressure(float* pressureField, size_t xSize, size_t ySize, int x, int y, float B, float alpha, float beta)
{
	float C = pressureField[int(y) * xSize + int(x)];
	float xU = C, xD = C, xL = C, xR = C;
#define SET(P, x, y) if (x < xSize && x >= 0 && y < ySize && y >= 0) P = pressureField[int(y) * xSize + int(x)]
	SET(xU, x, y - 1);
	SET(xD, x, y + 1);
	SET(xL, x - 1, y);
	SET(xR, x + 1, y);
#undef SET
	float pressure = (xU + xD + xL + xR + alpha * B) * (1.0f / beta);
	return pressure;
}

// performs iteration of jacobi method on color grid field
__device__ Color3f jacobiColor(Particle* colorField, size_t xSize, size_t ySize, Vec2 pos, Color3f B, float alpha, float beta)
{
	Color3f xU, xD, xL, xR, res;
	int x = pos.x;
	int y = pos.y;
#define SET(P, x, y) if (x < xSize && x >= 0 && y < ySize && y >= 0) P = colorField[int(y) * xSize + int(x)]
	SET(xU, x, y - 1).color;
	SET(xD, x, y + 1).color;
	SET(xL, x - 1, y).color;
	SET(xR, x + 1, y).color;
#undef SET
	res = (xU + xD + xL + xR + B * alpha) * (1.0f / beta);
	return res;
}

// interpolates quantity of grid cells
__device__ Particle interpolate(Vec2 v, Particle* field, size_t xSize, size_t ySize)
{
	float x1 = (int)v.x;
	float y1 = (int)v.y;
	float x2 = (int)v.x + 1;
	float y2 = (int)v.y + 1;
	Particle q1, q2, q3, q4;
#define CLAMP(val, minv, maxv) min(maxv, max(minv, val))
#define SET(Q, x, y) Q = field[int(CLAMP(y, 0.0f, ySize - 1.0f)) * xSize + int(CLAMP(x, 0.0f, xSize - 1.0f))]
	SET(q1, x1, y1);
	SET(q2, x1, y2);
	SET(q3, x2, y1);
	SET(q4, x2, y2);
#undef SET
#undef CLAMP
	float t1 = (x2 - v.x) / (x2 - x1);
	float t2 = (v.x - x1) / (x2 - x1);
	Vec2 f1 = q1.u * t1 + q3.u * t2;
	Vec2 f2 = q2.u * t1 + q4.u * t2;
	Color3f C1 = q2.color * t1 + q4.color * t2;
	Color3f C2 = q2.color * t1 + q4.color * t2;
	float t3 = (y2 - v.y) / (y2 - y1);
	float t4 = (v.y - y1) / (y2 - y1);
	Particle res;
	res.u = f1 * t3 + f2 * t4;
	res.color = C1 * t3 + C2 * t4;
	return res;
}

// computes gradient of pressure field
__device__ Vec2 gradient(float* field, size_t xSize, size_t ySize, int x, int y)
{
	float C = field[y * xSize + x];
#define SET(P, x, y) if (x < xSize && x >= 0 && y < ySize && y >= 0) P = field[int(y) * xSize + int(x)]
	float x1 = C, x2 = C, y1 = C, y2 = C;
	SET(x1, x + 1, y);
	SET(x2, x - 1, y);
	SET(y1, x, y + 1);
	SET(y2, x, y - 1);
#undef SET
	Vec2 res = { (x1 - x2) * 0.5f, (y1 - y2) * 0.5f };
	return res;
}

__global__ void computeVorticity(float* vField, Particle* field, size_t xSize, size_t ySize)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	vField[y * xSize + x] = curl(field, xSize, ySize, x, y);
}

// applies vorticity to velocity field
__global__ void applyVorticity(Particle* newField, Particle* oldField, float* vField, size_t xSize, size_t ySize, float vorticity, float dt)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	Particle& pOld = oldField[y * xSize + x];
	Particle& pNew = newField[y * xSize + x];

	Vec2 v = absGradient(vField, xSize, ySize, x, y);
	// v.y *= -1.0f; // 왜?

	float length = sqrtf(v.x * v.x + v.y * v.y) + 1e-5f;
	Vec2 vNorm = v * (1.0f / length);

	Vec2 vF = vNorm * vField[y * xSize + x] * vorticity;
	pNew = pOld;
	pNew.u = pNew.u + vF * dt;

}

// calculates nonzero divergency velocity field u
__global__ void diffuse(Particle* newField, Particle* oldField, size_t xSize, size_t ySize, float vDiffusion, float dt)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	Vec2 pos = { x * 1.0f, y * 1.0f };
	Vec2 u = oldField[y * xSize + x].u;

	// perfoms one iteration of jacobi method (diffuse method should be called 20-50 times per cell)
	float alpha = vDiffusion * vDiffusion / dt;
	float beta = 4.0f + alpha;
	newField[y * xSize + x].u = jacobiVelocity(oldField, xSize, ySize, pos, u, alpha, beta);
}

// calculates color field diffusion
__global__ void computeColor(Particle* newField, Particle* oldField, size_t xSize, size_t ySize, float cDiffusion, float dt)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	Vec2 pos = { x * 1.0f, y * 1.0f };
	Color3f c = oldField[y * xSize + x].color;
	float alpha = cDiffusion * cDiffusion / dt;
	float beta = 4.0f + alpha;
	// perfom one iteration of jacobi method (diffuse method should be called 20-50 times per cell)
	newField[y * xSize + x].color = jacobiColor(oldField, xSize, ySize, pos, c, alpha, beta);
}

// applies force and add color dye to the particle field
__global__ void applyForce(Particle* field, size_t xSize, size_t ySize, Color3f color, Vec2 F, Vec2 pos, int r, float dt)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	float e = expf(-((x - pos.x) * (x - pos.x) + (y - pos.y) * (y - pos.y)) / r);
	Vec2 uF = F * dt * e;
	Particle& p = field[y * xSize + x];
	p.u = p.u + uF;

	color = color * e + p.color;
	p.color.R = color.R;
	p.color.G = color.G;
	p.color.B = color.B;
}

// performs iteration of jacobi method on pressure field
__global__ void computePressureImpl(Particle* field, size_t xSize, size_t ySize, float* pNew, float* pOld, float pressure, float dt)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	float div = divergency(field, xSize, ySize, x, y);

	float alpha = -1.0f * pressure * pressure;
	float beta = 4.0;
	pNew[y * xSize + x] = jacobiPressure(pOld, xSize, ySize, x, y, div, alpha, beta);
}

// adds quantity to particles using bilinear interpolation
__global__ void advect(Particle* newField, Particle* oldField, size_t xSize, size_t ySize, float dDiffusion, float dt)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	float decay = 1.0f / (1.0f + dDiffusion * dt);
	Vec2 pos = { x * 1.0f, y * 1.0f };
	Particle& Pold = oldField[y * xSize + x];
	// find new particle tracing where it came from
	Particle p = interpolate(pos - Pold.u * dt, oldField, xSize, ySize);
	p.u = p.u * decay;
	p.color.R = min(1.0f, pow(p.color.R, 1.005f) * decay);
	p.color.G = min(1.0f, pow(p.color.G, 1.005f) * decay);
	p.color.B = min(1.0f, pow(p.color.B, 1.005f) * decay);
	newField[y * xSize + x] = p;
}

// projects pressure field on velocity field
__global__ void project(Particle* newField, size_t xSize, size_t ySize, float* pField)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	Vec2& u = newField[y * xSize + x].u;
	u = u - gradient(pField, xSize, ySize, x, y);
}

// fills output image with corresponding color
__global__ void paint(float4* colorField, Particle* field, size_t xSize, size_t ySize)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	float R = field[y * xSize + x].color.R;
	float G = field[y * xSize + x].color.G;
	float B = field[y * xSize + x].color.B;

	colorField[y * xSize + x] = make_float4(min(255.0f, 255.0f * R), min(255.0f, 255.0f * G), min(255.0f, 255.0f * B), 255.0f); // make_float4(min(255.0f, 255.0f * R), min(255.0f, 255.0f * G), min(255.0f, 255.0f * B), 255.0f);
}

// main function, calls vorticity -> diffusion -> apply force -> pressure -> project -> advect -> paint -> bloom
void PhysicsContainer::computeField(float dt, float f_Anim, int x2pos, int y2pos, bool isPressed)
{
    // execute the kernel
    dim3 block(sConfig.xThreads, sConfig.yThreads, 1);
    dim3 grid(xSize / block.x, ySize / block.y, 1);

    // curls and vortisity
    computeVorticity<<<grid, block>>>(this->vorticityField, this->oldField, xSize, ySize);
    applyVorticity<<<grid, block>>>(this->newField, this->oldField, vorticityField, xSize, ySize, config.vorticity, dt);
    std::swap(oldField, newField);

    // diffuse velocity [and color]
	for (int i = 0; i < sConfig.velocityIterations; i++)
	{
		diffuse<<<grid, block>>>(this->newField, this->oldField, xSize, ySize, config.velocityDiffusion, dt);
		computeColor<<<grid, block>> > (this->newField, this->oldField, xSize, ySize, config.colorDiffusion, dt);
		std::swap(newField, oldField);
	}

    // apply force
    if(isPressed)
    {
    	timeSincePress = 0.0f;
    	elapsedTime += dt;
		// apply gradient to color
		int roundT = int(elapsedTime) % colorArraySize;
		int ceilT = int((elapsedTime)+1) % colorArraySize;
		float w = elapsedTime - int(elapsedTime);
		currentColor = colorArray[roundT] * (1 - w) + colorArray[ceilT] * w;

    	Vec2 F; // 외부에서 가해지는 힘
    	float scale = config.forceScale * 10.0f;
    	F.x = scale * cos(f_Anim * sin(f_Anim / 2)); // 디폴트 힘
    	F.y = scale * sin(f_Anim * sin(f_Anim / 2)); // 디폴트 힘
    	Vec2 pos = { x2pos + (cos(f_Anim) - sin(f_Anim)) * 100.0f, y2pos + (sin(f_Anim) + cos(f_Anim)) * 100.0f }; // 힘이 가해진 위치

    	applyForce<<<grid, block>>>(this->oldField, xSize, ySize, currentColor, F, pos, config.radius, dt);
    }
    else
    {
    	timeSincePress += dt;
    }

    // compute pressure
	for (int i = 0; i < sConfig.pressureIterations; i++)
	{
		computePressureImpl <<<grid, block>>>(this->oldField, xSize, ySize, this->pressureNew, this->pressureOld, config.pressure, dt);
		std::swap(pressureOld, pressureNew);
	}

	// project
	project<<<grid, block>>>(this->oldField, xSize, ySize, this->pressureOld);
	cudaMemset(pressureOld, 0, xSize * ySize * sizeof(float));

	// advect
	advect<<<grid, block>>>(this->newField, this->oldField, xSize, ySize, config.densityDiffusion, dt);
	std::swap(newField, oldField);

	// paint image
	paint<<<grid, block>>>(this->colorField, this->oldField, xSize, ySize);

}




