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
// Nev    : Gabor Klemm
// Neptun : H8XK58
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

inline vec4 pToV(vec4 point){
	return point.w > 0 ? point-vec4(0,0,0,point.w) : point;
}

const char * const vertexSource = R"(
	#version 330

	float dot(vec4 v0, vec4 v1){
		return v0.x*v1.x + v0.y*v1.y + v0.z*v1.z + v0.w*v1.w;
	}
	

	uniform vec4 rayRad;
	uniform vec4 ambientRad;
	uniform mat4 globalTransf;
	// expect that these vectors are normalized !!
	uniform vec4 rayDir;

	layout(location = 0) in vec4 vertexPos;
	layout(location = 1) in vec4 vertexNormal;

	out vec4 radiance;

	void main() {
		gl_Position = vec4(vertexPos.x, vertexPos.y, vertexPos.z, 1) * globalTransf;
		radiance = ambientRad;
		if(dot(rayDir, vertexNormal*globalTransf) < 0.02f) radiance += rayRad;	
	}
)";

// fragment shader in GLSL
const char * const fragmentSource = R"(
	#version 330

	
	in vec4 radiance;
	out vec4 outColor;

	void main() {
		outColor = radiance;
	}
)";
GPUProgram gpuProgram; // vertex and fragment shaders
class Object3D{
	unsigned int vao, vbo;

protected:
	struct VertexData{
		vec4 pos, normal;
		VertexData(vec4 _pos, vec4 _normal) : pos(_pos), normal(_normal){} 
	};
	std::vector<VertexData> vertices;
	mat4 globalTransf = { 1.0f, 0.0f, 0.0f, 0.0f,
				0.0f, 1.0f, 0.0f, 0.0f,
				0.0f, 0.0f, 1.0f, 0.0f,
				0.0f, 0.0f, 0.0f, 1.0f };

	virtual void tescellate() = 0;
public:
	void create(){
		glGenVertexArrays(1, &vao);
		glGenBuffers(1, &vbo);
		glBindVertexArray(vao);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);

		glBufferData(GL_ARRAY_BUFFER, vertices.size()*sizeof(VertexData), &vertices[0], GL_STATIC_DRAW);

		glEnableVertexAttribArray(0);
		glEnableVertexAttribArray(1);

		glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, NULL);
		glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, NULL);


	}
	void draw(){
		
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		gpuProgram.setUniform(globalTransf, "globalTransf");
		glDrawArrays(GL_TRIANGLE_STRIP, 0, vertices.size());
	}
	void applyMtx(mat4 m){
		globalTransf = m*globalTransf;
	}
};

class TestTriangle : public Object3D{
	vec4 v0, v1, v2;
public: 
	TestTriangle(vec4 _v0, vec4 _v1, vec4 _v2) : v0(_v0), v1(_v1), v2(_v2){
		tescellate();
	}
	void tescellate() override{
		vec3 normal = cross(vec3(v0.x, v0.y, v0.z), vec3(v1.x, v1.y, v1.z));
		vertices.push_back(VertexData(v0, {normal.x, normal.y, normal.z, 0} ));
		vertices.push_back(VertexData(v1, {normal.x, normal.y, normal.z, 0} ));
		vertices.push_back(VertexData(v2, {normal.x, normal.y, normal.z, 0} ));
	}
	
};


TestTriangle t = {{1, 1, 1}, {1, 0, 0}, {0, 0, 1}};

void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);

	
	t.create();
	t.applyMtx(TranslateMatrix({-0.5, -0.5, 0}));
	gpuProgram.create(vertexSource, fragmentSource, "outColor");
	gpuProgram.setUniform(vec4(0.2f, 0.2f, 0.2f, 1.0f), "ambientRad");
	gpuProgram.setUniform(vec4(1.0f, 1.0f, 1.0f, 1.0f), "rayRad");
	gpuProgram.setUniform(vec4(0.0f, 0.0f, -1.0f, 0.0f), "rayDir");
}


void onDisplay() {
	glClearColor(0, 0, 0, 0);   
	glClear(GL_COLOR_BUFFER_BIT);
	t.draw();
	glutSwapBuffers(); 
}

void onKeyboard(unsigned char key, int pX, int pY) {
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {	// pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system

}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) {
	float cX = 2.0f * pX / windowWidth - 1;	
	float cY = 1.0f - 2.0f * pY / windowHeight;  
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
	t.applyMtx(RotationMatrix(0.01, {1, 1, 1}));
	glutPostRedisplay();
}
