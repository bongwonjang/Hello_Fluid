#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <math.h>
#include <assert.h>

#include "GL/glew.h"
#include "GL/freeglut.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>
#include <device_launch_parameters.h>
#include <vector_types.h>

#include "header/ErrorCheck.cuh"
#include "header/Physics.cuh"

#define MAX_EPSILON_ERROR	10.0f
#define THRESHOLD			0.30f
#define REFRESH_DELAY		10 // ms

////////////////////////////////////////////////////////////////////////////////
// constants
const unsigned int window_width  = 512;
const unsigned int window_height = 512;

// Texture variables
GLuint texture_id;
struct cudaResourceDesc cuda_resourceDesc;
struct cudaGraphicsResource *cuda_texture_resource;
cudaArray_t cuda_texture_array;
cudaSurfaceObject_t cuda_surfaceObj = 0;

// mouse controls
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;
float translate_z = -3.0;

// Physics Container
PhysicsContainer* pc = nullptr;
float f_Anim = 0.0f;

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
bool runTest(int argc, char **argv);
void cleanup();

// GL functionality
bool initGL(int *argc, char **argv);
GLuint createTexture(struct cudaGraphicsResource **texture_res, unsigned int texture_res_flags);
void deleteTexture(GLuint texture, struct cudaGraphicsResource *texture_res);

// rendering callbacks
void display();
void keyboard(unsigned char key, int x, int y);
void mouse(int button, int state, int x, int y);
void motion(int x, int y);
void timerEvent(int value);

// Cuda functionality
void runCuda(struct cudaGraphicsResource **texture_resource);
void checkResultCuda(int argc, char **argv, const GLuint &vbo);

const char *sSDKsample = "simpleGL (VBO)";

///////////////////////////////////////////////////////////////////////////////
//! Simple kernel to modify vertex positions in sine wave pattern
//! @param data  data in global memory
///////////////////////////////////////////////////////////////////////////////
__global__ void render_texture_kernel(cudaSurfaceObject_t surface, float4* colorField, unsigned int width, unsigned int height)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x < width && y < height)
    {
    	float4 color = colorField[y * width + x];
    	uchar4 pixel_data = make_uchar4((int)color.x, (int)color.y, (int)color.z, 255);

    	// Texture Memory에다가 직접 값을 넣는 방법
    	surf2Dwrite(pixel_data, surface, x * sizeof(uchar4), y);
    }
}

void launch_kernel(cudaSurfaceObject_t surface, unsigned int window_width,
                   unsigned int window_height)
{
	if(pc == nullptr)
	{
		printf("PhysicsContainer is NULL\n");
		return;
	}

	float dt = 0.01f;
	pc->computeField(dt, f_Anim, window_width / 2, window_height / 2, true);

    // execute the kernel
    dim3 block(32, 32, 1);
    dim3 grid(window_width / block.x, window_height / block.y, 1);

    render_texture_kernel<<< grid, block>>>(surface, pc->colorField, window_width, window_height);
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
#if defined(__linux__)
    setenv ("DISPLAY", ":0", 0);
#endif

    pc = new PhysicsContainer(window_width, window_height);

    printf("%s starting...\n", sSDKsample);

    runTest(argc, argv);
}

////////////////////////////////////////////////////////////////////////////////
//! Initialize GL
////////////////////////////////////////////////////////////////////////////////
bool initGL(int *argc, char **argv)
{
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(window_width, window_height);
    glutCreateWindow("Cuda GL Interop (VBO)");

    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
    glutTimerFunc(REFRESH_DELAY, timerEvent,0);

    // initialize necessary OpenGL extensions
    // VBO를 사용하기 위해선 필수(내가 알고 있기로는)라서, glewInit()이 없으면, 프로그램 실행이 안 됨
    glewInit();

    // default initialization
    glClearColor(0.0, 0.0, 0.0, 1.0);
    glDisable(GL_DEPTH_TEST);

    // viewport
    glViewport(0, 0, window_width, window_height);

    // projection
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, window_width, 0, window_height, -1, 1);

    return true;
}

////////////////////////////////////////////////////////////////////////////////
//! Create Texture
////////////////////////////////////////////////////////////////////////////////
GLuint createTexture(struct cudaGraphicsResource **texture_res, unsigned int texture_res_flags)
{
	GLuint texture = 0;

	glGenTextures(1, &texture);
	glBindTexture(GL_TEXTURE_2D, texture);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, window_width, window_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

	glBindTexture(GL_TEXTURE_2D, 0);

	gpuErrchk(cudaGraphicsGLRegisterImage(texture_res, texture, GL_TEXTURE_2D, texture_res_flags));

	// resource description for surface
	memset(&cuda_resourceDesc, 0, sizeof(cuda_resourceDesc));
	cuda_resourceDesc.resType = cudaResourceTypeArray;

	return texture;
}

////////////////////////////////////////////////////////////////////////////////
//! Delete Texture
////////////////////////////////////////////////////////////////////////////////
void deleteTexture(GLuint texture, struct cudaGraphicsResource *texture_res)
{
    // unregister this buffer object with CUDA
	gpuErrchk(cudaGraphicsUnregisterResource(texture_res));

	glBindTexture(GL_TEXTURE_2D, texture);
	glDeleteTextures(1, &texture);
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
bool runTest(int argc, char **argv)
{
    // Initialize OpenGL settings
	initGL(&argc, argv);

#ifdef __APPLE__
  cudaError_t cError = cudaGLSetGLDevice(0); // Linux와 Window에선 deprecated된 함수
#else
  cudaError_t cError = cudaSetDevice(0);
#endif

#if defined (__APPLE__) || defined(MACOSX)
     atexit(cleanup);
#else
	glutCloseFunc(cleanup);
#endif

	// create Field
	pc->cudaInit();

	// create Texture
	texture_id = createTexture(&cuda_texture_resource, cudaGraphicsRegisterFlagsSurfaceLoadStore);

	// start rendering main-loop
	glutMainLoop();

    return true;
}

////////////////////////////////////////////////////////////////////////////////
//! Run the Cuda part of the computation
////////////////////////////////////////////////////////////////////////////////
void runCuda(struct cudaGraphicsResource **texture_resource)
{
    // map OpenGL buffer object for writing from CUDA
	// https://stackoverflow.com/questions/9406844/cudagraphicsresourcegetmappedpointer-returns-unknown-error
	/*
	 * CUDA 연산을 할 때는, 이렇게 MapResources 방식으로 진행하는 군!
	 * 즉, 연산과 렌더링을 구분하여 아래와 같이 진행.
	 * CUDA 연산 -> cudaGraphicsMap ~~~
	 * OpenGL 렌더링 -> BindBuffer ~~~
	 */
    gpuErrchk(cudaGraphicsMapResources(1, texture_resource, 0));
    gpuErrchk(cudaGraphicsSubResourceGetMappedArray(&cuda_texture_array, *texture_resource, 0, 0));

    cuda_resourceDesc.res.array.array = cuda_texture_array;
    gpuErrchk(cudaCreateSurfaceObject(&cuda_surfaceObj, &cuda_resourceDesc));

    launch_kernel(cuda_surfaceObj, window_width, window_height);

    // unmap buffer object
    /*
     * 연산을 끝냈다면, Unmap!
     */
    gpuErrchk(cudaGraphicsUnmapResources(1, texture_resource, 0));
}

////////////////////////////////////////////////////////////////////////////////
//! Display callback
////////////////////////////////////////////////////////////////////////////////
void display()
{
    // run CUDA kernel to generate vertex positions
	runCuda(&cuda_texture_resource);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // set view matrix
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    // render using texture
    glBindTexture(GL_TEXTURE_2D, texture_id);
    glEnable(GL_TEXTURE_2D);
    glBegin(GL_QUADS);
    	glTexCoord2i(0, 0); glVertex2i(0, 0);
    	glTexCoord2i(0, 1); glVertex2i(0, window_height);
    	glTexCoord2i(1, 1); glVertex2i(window_width, window_height);
    	glTexCoord2i(1, 0); glVertex2i(window_width, 0);
    glEnd();
    glDisable(GL_TEXTURE_2D);

    glutSwapBuffers();
}

void timerEvent(int value)
{
    if (glutGetWindow())
    {
        glutPostRedisplay();
        glutTimerFunc(REFRESH_DELAY, timerEvent,0);
    }

    f_Anim += 0.1f;
}

void cleanup()
{
    if (texture_id)
    {
    	deleteTexture(texture_id, cuda_texture_resource);
    }

    if(pc != nullptr)
    {
    	pc->cudaExit();

    	delete pc;
    	pc = nullptr;
    }

    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    cudaDeviceReset();
}


////////////////////////////////////////////////////////////////////////////////
//! Keyboard events handler
////////////////////////////////////////////////////////////////////////////////
void keyboard(unsigned char key, int /*x*/, int /*y*/)
{
    switch (key)
    {
        case (27) :
            #if defined(__APPLE__) || defined(MACOSX)
                exit(EXIT_SUCCESS);
            #else
                glutDestroyWindow(glutGetWindow());
                return;
            #endif
    }
}

////////////////////////////////////////////////////////////////////////////////
//! Mouse event handlers
////////////////////////////////////////////////////////////////////////////////
void mouse(int button, int state, int x, int y)
{
    if (state == GLUT_DOWN)
    {
        mouse_buttons |= 1<<button;
    }
    else if (state == GLUT_UP)
    {
        mouse_buttons = 0;
    }

    mouse_old_x = x;
    mouse_old_y = y;
}

void motion(int x, int y)
{
    float dx, dy;
    dx = (float)(x - mouse_old_x);
    dy = (float)(y - mouse_old_y);

    if (mouse_buttons & 1)
    {
        rotate_x += dy * 0.2f;
        rotate_y += dx * 0.2f;
    }
    else if (mouse_buttons & 4)
    {
        translate_z += dy * 0.01f;
    }

    mouse_old_x = x;
    mouse_old_y = y;
}
