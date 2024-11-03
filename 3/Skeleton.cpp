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

const int mapBytes[] = {252, 252, 252, 252, 252, 252, 252, 252, 252, 0, 9, 80, 1, 148, 13, 72, 13, 140, 25, 60, 21, 132, 41, 12, 1, 28,
25, 128, 61, 0, 17, 4, 29, 124, 81, 8, 37, 116, 89, 0, 69, 16, 5, 48, 97, 0, 77, 0, 25, 8, 1, 8, 253, 253, 253, 253,
101, 10, 237, 14, 237, 14, 241, 10, 141, 2, 93, 14, 121, 2, 5, 6, 93, 14, 49, 6, 57, 26, 89, 18, 41, 10, 57, 26,
89, 18, 41, 14, 1, 2, 45, 26, 89, 26, 33, 18, 57, 14, 93, 26, 33, 18, 57, 10, 93, 18, 5, 2, 33, 18, 41, 2, 5, 2, 5, 6,
89, 22, 29, 2, 1, 22, 37, 2, 1, 6, 1, 2, 97, 22, 29, 38, 45, 2, 97, 10, 1, 2, 37, 42, 17, 2, 13, 2, 5, 2, 89, 10, 49,
46, 25, 10, 101, 2, 5, 6, 37, 50, 9, 30, 89, 10, 9, 2, 37, 50, 5, 38, 81, 26, 45, 22, 17, 54, 77, 30, 41, 22, 17, 58,
1, 2, 61, 38, 65, 2, 9, 58, 69, 46, 37, 6, 1, 10, 9, 62, 65, 38, 5, 2, 33, 102, 57, 54, 33, 102, 57, 30, 1, 14, 33, 2,
9, 86, 9, 2, 21, 6, 13, 26, 5, 6, 53, 94, 29, 26, 1, 22, 29, 0, 29, 98, 5, 14, 9, 46, 1, 2, 5, 6, 5, 2, 0, 13, 0, 13,
118, 1, 2, 1, 42, 1, 4, 5, 6, 5, 2, 4, 33, 78, 1, 6, 1, 6, 1, 10, 5, 34, 1, 20, 2, 9, 2, 12, 25, 14, 5, 30, 1, 54, 13, 6,
9, 2, 1, 32, 13, 8, 37, 2, 13, 2, 1, 70, 49, 28, 13, 16, 53, 2, 1, 46, 1, 2, 1, 2, 53, 28, 17, 16, 57, 14, 1, 18, 1, 14,
1, 2, 57, 24, 13, 20, 57, 0, 2, 1, 2, 17, 0, 17, 2, 61, 0, 5, 16, 1, 28, 25, 0, 41, 2, 117, 56, 25, 0, 33, 2, 1, 2, 117,
52, 201, 48, 77, 0, 121, 40, 1, 0, 205, 8, 1, 0, 1, 12, 213, 4, 13, 12, 253, 253, 253, 141};

const vec4 colors[] = {{1,1,1,1},{0,0,1,1},{0,1,0,1},{0,0,0,1}};

const int MAP_WIDTH = 64;
const int MAP_HEIGHT = 64;
const float GLOBE_CIRCUMFERENCE = 40000;
const float GLOBE_RADIUS = GLOBE_CIRCUMFERENCE / M_2_PI;
// vertex shader in GLSL: It is a Raw string (C++11) since it contains new line characters
const char * const vertexSource = R"(
	#version 330

	uniform mat4 MVP;

	layout(location = 0) in vec2 vertexPos;
	layout(location = 1) in vec2 vertexUV;

	out vec2 texCoord;

	void main() {
		texCoord = vertexUV;
		gl_Position = vec4(vertexPos.x, vertexPos.y, 0, 1) * MVP;
	}
)";

// fragment shader in GLSL
const char * const fragmentSource = R"(
	#version 330

	uniform int isBackground;
	
	uniform vec3 color;
	uniform sampler2D textureUnit;

	in vec2 texCoord;
	out vec4 outColor;

	void main() {
		if(isBackground != 0) { outColor = texture(textureUnit, texCoord); }
		else { outColor = vec4(color, 1); }
	}
)";

GPUProgram gpuProgram; // vertex and fragment shaders
unsigned int vao;	   // virtual world on the GPU

class Background{
	unsigned int vbo[2];
	vec2 vertices[4], uvs[4];
	Texture texture;

	void createBgTexture(){
		vec4 rgb;
		std::vector<vec4> image;
		for(int byte : mapBytes){
			rgb = colors[byte%4];
			for(int n = 0; n <= byte>>2; n++){
				image.push_back(rgb);
			}
		}
		texture.create(MAP_WIDTH, MAP_HEIGHT, image, GL_NEAREST);
	}
public:
	Background(){
		vertices[0] = { -1, -1};
		vertices[1] = { 1, -1};
		vertices[2] = { 1, 1};
		vertices[3] = { -1, 1};
		uvs[0] = {0, 0};
		uvs[1] = {1, 0};
		uvs[2] = {1, 1};
		uvs[3] = {0, 1};
	}

	void create(){
		createBgTexture();
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);

		glGenBuffers(2, vbo);

	}

	void updateGPU(){
		glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_DYNAMIC_DRAW);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);

		glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);
		glBufferData(GL_ARRAY_BUFFER, sizeof(uvs), uvs, GL_STATIC_DRAW);
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, NULL);
	}
	void draw(){
		updateGPU();
		glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER ,GL_NEAREST);
		
		gpuProgram.setUniform((int)true, "isBackground");
		gpuProgram.setUniform(texture, "textureUnit");
		gpuProgram.setUniform(vec3{1,1,1}, "color");
		gpuProgram.setUniform({1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1}, "MVP");
		glBindVertexArray(vao);
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
	}

};

class Slerp{
	unsigned int vbo;
	std::vector<vec2> points;
	std::vector<vec2> slice;
	const float sliceAmount = 100.0f;
	const float rad85deg = 1.48353f;

	float zScale(){
		return std::log(std::tan(M_PI_4 + rad85deg/2));
	}

	vec2 mercatorInv(vec2 v){
		return { v.x*M_PI, 2*(std::atan(std::exp(v.y*zScale()))-M_PI_4) };
	}
	vec2 mercator(vec2 v){
		return { v.x/M_PI, std::log(std::tan(M_PI_4 + (v.y)/2.0f))/zScale()};
	}
	
	//https://www.reddit.com/r/askmath/comments/ybbn3b/how_to_transform_latlong_coordinates_on_a_sphere/
	vec3 sphericalInv(vec2 v){
		return { std::cos(v.y)*std::sin(v.x), std::cos(v.y)*std::cos(v.x), std::sin(v.y)};
	}
	vec2 spherical(vec3 v){
		return {std::atan2(v.x, v.y), std::atan2(v.z, std::sqrt(v.x*v.x+v.y*v.y))};
	}
	void calculateSlerpPoints(vec2 start, vec2 end){
		vec2 mStart = mercatorInv(start);
		vec2 mEnd = mercatorInv(end);
		vec3 euclStart = sphericalInv(mStart);
		vec3 euclEnd = sphericalInv(mEnd);
		float theta = std::acos(dot(euclStart, euclEnd));
		float dist = theta*GLOBE_RADIUS;
		printf("Distance: %d km\n", (int)dist);
		for(float t = 0; t <= 1; t+=1/sliceAmount){
			vec3 slerpPoint = euclStart*((std::sin((1-t)*theta))/(std::sin(theta))) + euclEnd*((std::sin(t*theta))/(std::sin(theta)));
			slice.push_back(mercator(spherical(slerpPoint)));
		}
	}
public:
	void create(){
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);
		glGenBuffers(1, &vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		
	}

	void addPoint(float cX, float cY){
		points.push_back({cX, cY});
		if(points.size() > 1){
			calculateSlerpPoints(*(points.end()-2), points.back());
		}
	}
	void updateGPU(){
		glBindVertexArray(vao);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);
	}
	void draw(){
		updateGPU();
		gpuProgram.setUniform((int)false, "isBackground");
		gpuProgram.setUniform({1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1}, "MVP");
		if(slice.size() > 1){
			gpuProgram.setUniform(vec3(1,1,0), "color");
			
			glBufferData(GL_ARRAY_BUFFER, sizeof(vec2)*slice.size(), &slice[0], GL_DYNAMIC_DRAW);
			glDrawArrays(GL_LINE_STRIP, 0, slice.size());
		}
		gpuProgram.setUniform(vec3(1,0,0), "color");
		glBufferData(GL_ARRAY_BUFFER, sizeof(vec2)*points.size(), &points[0], GL_DYNAMIC_DRAW);
		glDrawArrays(GL_POINTS, 0, points.size());
	}

};

Background bg;
Slerp slerp;
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	
	glPointSize(10);
	glLineWidth(3);
	bg.create();
	slerp.create();
	gpuProgram.create(vertexSource, fragmentSource, "outColor");
}


void onDisplay() {
	glClearColor(0, 0, 0, 0);   
	glClear(GL_COLOR_BUFFER_BIT); 

	bg.draw();
	slerp.draw();
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
	if(state == GLUT_DOWN && button == GLUT_LEFT){
		slerp.addPoint(cX, cY);	
		glutPostRedisplay();
	}
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
}
