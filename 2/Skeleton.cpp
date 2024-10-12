//=============================================================================================
// Mintaprogram: Z�ld h�romsz�g. Ervenyes 2019. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : 
// Neptun : 
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"

// vertex shader in GLSL: It is a Raw string (C++11) since it contains new line characters
const char * const vertexSource = R"(
	#version 330
	precision highp float;		// normal floats, makes no difference on desktop computers

	uniform mat4 MVP;			// uniform variable, the Model-View-Projection transformation matrix
	layout(location = 0) in vec3 vp;	// Varying input: vp = vertex position is expected in attrib array 0

	void main() {
		gl_Position = vec4(vp.x, vp.y, vp.z, 1) * MVP;		// transform vp from modeling space to normalized device space
	}
)";

// fragment shader in GLSL
const char * const fragmentSource = R"(
	#version 330
	precision highp float;	// normal floats, makes no difference on desktop computers
	
	uniform vec3 color;		// uniform variable, the color of the primitive
	out vec4 outColor;		// computed color of the current pixel

	void main() {
		outColor = vec4(color, 1);	// computed color is the color of the primitive
	}
)";

GPUProgram gpuProgram; // vertex and fragment shaders
unsigned int vao;	   // virtual world on the GPU

class Object {
	unsigned int vao, vbo;
	std::vector<vec3> vertices;

public:
	void create() {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);
		glGenBuffers(1, &vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	}

	void load(vec3 v){
		vertices.push_back(v);
	}
	void transform(mat4 m) {
		for(vec3& v : vertices) {
			vec4 v4 = { v.x, v.y, 1.0f, 1.0f };
			v4 = v4*m;
			v = { v4.x, v4.y, 1.0f};
		}
		updateGPU();
	}
	void updateGPU(){
		glBindVertexArray(vao);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, vertices.size()*sizeof(vec3), &vertices[0], GL_DYNAMIC_DRAW);
	}

	void draw(GLenum mode, vec3 color) {
		updateGPU();
		gpuProgram.setUniform(color, "color");
		glDrawArrays(mode, 0, vertices.size());
	}
	
};
class Ball {
	Object obj;
	vec3 centre;
	float r;

public:
	void create() {
		r = 0.075f;
		obj.create();
		float nRot = 15;
		vec4 v = { 0.0f, r, 1.0f, 1.0f};
		mat4 rot = RotationMatrix(M_PI*2/nRot, {0, 0, 1});
		for(float f = 0; f <= M_PI*2; f+=M_PI*2/nRot){
			obj.load({v.x, v.y, 1});
			v=v*rot;
		}
		obj.load({ 0.0f, r, 1.0f });
		obj.load({ 0.0f, -r, 1.0f });
		obj.load({ 0.0f, 0.0f, 1.0f });
		obj.load({ r, 0.0f, 1.0f });
		obj.load({ -r, 0.0f, 1.0f });
		obj.updateGPU();
	}

	void draw() {
		obj.draw(GL_TRIANGLE_FAN, { 0.0f, 0.0f, 1.0f });
		obj.draw(GL_LINE_STRIP, { 1.0f, 1.0f, 1.0f });
	}
	void rotate(float phi) {
		mat4 rotation = RotationMatrix(phi, { 0.0f, 0.0f, 1.0f });
		obj.transform(rotation);
	}
};

class CRSpline {
	Object spline;
	std::vector<vec3> cpoints;
	std::vector<int> params;
	vec3 hermite(vec3 p0, vec3 p1, vec3 v0, vec3 v1, float t0, float t1, float t) {

	}

public:
	void create(){
		spline.create();
	}
	void addCPoint(vec3 pos) {
		if(params.size() == 0) params.push_back(0);
		else { params.push_back(params[params.size()-1] + 1); }
		cpoints.push_back(pos);
		spline.load(pos);
	}
	vec3 expandSpline(){
		// vec3 vi, vi1;
		// vi = vi1 = {0, 0, 1};
		// for(int i = 0; i < cpoints.size() - 1; i++) {
		// 	spline.load(hermite({ cpoints[i].x, cpoints[i].y, 1.0f }, 
		// 				{ cpoints[i+1].x, cpoints[i+1].y, 1.0f },
		// 				));
		// 	vi = 1/2*(()/() + ()/())
		// }
		
	}
	void draw() {
		spline.draw(GL_LINE_STRIP, { 1.0f, 1.0f, 0.0f });
		spline.draw(GL_POINTS, { 1.0f, 0.0f, 0.0f });
	}
};
CRSpline spline;
Ball ball;

void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	glPointSize(10);
	ball.create();
	spline.create();
	spline.addCPoint({0, 0, 1});
	spline.addCPoint({0.34, -0.54, 1});
	spline.addCPoint({-0.2234, -0.1123, 1});
	//spline.expandSpline();

	gpuProgram.create(vertexSource, fragmentSource, "outColor");
}
// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0, 0, 0, 0);     // background color
	glClear(GL_COLOR_BUFFER_BIT); // clear frame buffer

	// Set color to (0, 1, 0) = green
	int location = glGetUniformLocation(gpuProgram.getId(), "color");
	glUniform3f(location, 0.0f, 1.0f, 0.0f); // 3 floats

	float MVPtransf[4][4] = { 1, 0, 0, 0,    // MVP matrix, 
							  0, 1, 0, 0,    // row-major!
							  0, 0, 1, 0,
							  0, 0, 0, 1 };

	location = glGetUniformLocation(gpuProgram.getId(), "MVP");	// Get the GPU location of uniform variable MVP
	glUniformMatrix4fv(location, 1, GL_TRUE, &MVPtransf[0][0]);	// Load a 4x4 row-major float matrix to the specified location

	
	ball.draw();
	spline.draw();
	glutSwapBuffers(); // exchange buffers for double buffering
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'd') glutPostRedisplay();         // if d, invalidate display, i.e. redraw
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {	// pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;
	printf("Mouse moved to (%3.2f, %3.2f)\n", cX, cY);
}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) { // pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;

	char * buttonStat;
	switch (state) {
	case GLUT_DOWN: buttonStat = "pressed"; break;
	case GLUT_UP:   buttonStat = "released"; break;
	}

	switch (button) {
	case GLUT_LEFT_BUTTON:   printf("Left button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY);   break;
	case GLUT_MIDDLE_BUTTON: printf("Middle button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY); break;
	case GLUT_RIGHT_BUTTON:  printf("Right button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY);  break;
	}
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	ball.rotate(-0.05);
	glutPostRedisplay();
}
