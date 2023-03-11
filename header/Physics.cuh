/*
 * Physics.cuh
 *
 *  Created on: Mar 10, 2023
 *      Author: bongwon
 */

#ifndef HEADER_PHYSICS_CUH_
#define HEADER_PHYSICS_CUH_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <math.h>
#include <assert.h>
#include <utility>

#include "GL/glew.h"
#include "GL/freeglut.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>
#include <device_launch_parameters.h>
#include <vector_types.h>

#include "./ErrorCheck.cuh"

struct Vec2
{
	float x = 0.0, y = 0.0;

	__device__ Vec2 operator-(Vec2 other)
	{
		Vec2 res;
		res.x = this->x - other.x;
		res.y = this->y - other.y;
		return res;
	}

	__device__ Vec2 operator+(Vec2 other)
	{
		Vec2 res;
		res.x = this->x + other.x;
		res.y = this->y + other.y;
		return res;
	}

	__device__ Vec2 operator*(float other)
	{
		Vec2 res;
		res.x = this->x * other;
		res.y = this->y * other;
		return res;
	}
};

struct Color3f
{
	float R = 0.0f;
	float G = 0.0f;
	float B = 0.0f;

	__host__ __device__ Color3f operator+ (Color3f other)
	{
		Color3f res;
		res.R = this->R + other.R;
		res.G = this->G + other.G;
		res.B = this->B + other.B;
		return res;
	}

	__host__ __device__ Color3f operator* (float d)
	{
		Color3f res;
		res.R = this->R * d;
		res.G = this->G * d;
		res.B = this->B * d;
		return res;
	}
};

struct Particle // Cell
{
	Vec2 u; // velocity
	Color3f color;
};

static struct Config
{
	float velocityDiffusion;
	float pressure;
	float vorticity;
	float colorDiffusion;
	float densityDiffusion;
	float forceScale;
	float bloomIntense;
	int radius;
	bool bloomEnabled;
} config;

static struct SystemConfig
{
	int velocityIterations = 8;
	int pressureIterations = 8;
	int xThreads = 32;
	int yThreads = 32;
} sConfig;

class PhysicsContainer
{
public:
	static const int colorArraySize = 7;
	Color3f colorArray[colorArraySize];

	Particle* oldField = nullptr;
	Particle* newField = nullptr;
	size_t xSize = 0, ySize = 0;
	float4* colorField = nullptr;
	float* pressureOld = nullptr;
	float* pressureNew = nullptr;
	float* vorticityField = nullptr;
	Color3f currentColor = {0.0f, 0.0f, 0.0f};
	float elapsedTime = 0.0f;
	float timeSincePress = 0.0f;

	PhysicsContainer();
	PhysicsContainer(int p_xSize, int p_ySize);
	~PhysicsContainer();

	void cudaInit();
	void cudaExit();

	void computeField(float dt, float f_Anim, int x2pos, int y2pos, bool isPressed);
};



#endif /* HEADER_PHYSICS_CUH_ */
