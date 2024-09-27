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
// Nev    : Klemm Gabor
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

float pyth2d(float a, float b){
	return std::sqrt(a*a + b*b);
}
// vertex shader in GLSL: It is a Raw string (C++11) since it contains new line characters
const char * const vertexSource = R"(
	#version 330
	layout(location = 0) in vec3 vertexPosition;
	void main() {
		gl_Position = vec4(vertexPosition, 1); // in NDC
	}
)";

// fragment shader in GLSL
const char * const fragmentSource = R"(
	#version 330
	uniform vec3 color;
	out vec4 fragmentColor;
	void main() {
		fragmentColor = vec4(color, 1);
	}
)";

GPUProgram gpuProgram; // vertex and fragment shaders
//unsigned int vao;	   // virtual world on the GPU

float clamp(float n, float max, float min) {
	if(n < min) return min;
	else if(n > max) return max;
	else return n;
}

bool insideBoundary(float n, float max, float min) {
	return n > min && n < max;
}

class Object{
	unsigned int vao, vbo; 
	std::vector<vec3> vertices;
public:
	void create(){
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);
		glGenBuffers(1, &vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	}

	std::vector<vec3>& getVertices() {
		return vertices;
	}

	void updateGPU(){
		glBindVertexArray(vao);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(vec3), &vertices[0], GL_DYNAMIC_DRAW);
	}

	void draw(int type, vec3 color) {
		if(vertices.size() > 0) {
			glBindVertexArray(vao);
			gpuProgram.setUniform(color, "color");
			glDrawArrays(type, 0, vertices.size());
		}
	}
};

class PointCollection{
	Object points;
	size_t pointCount;
	
public:
	bool createPoints;
	void create(){
		createPoints = false;
		points.create();
		pointCount = 0;
	}
	void addPoint(vec3 pos){
		pointCount++;
		printf("%f, %f\n", pos.x, pos.y);
		points.getVertices().push_back(pos);
		points.updateGPU();
	}

	vec3* findNearest(float mX, float mY){
		vec3* nearest = &points.getVertices()[0];
		float minDist = pyth2d(mX - nearest->x, mY - nearest->y);

		for(int i = 0; i < pointCount; i++){
			vec3* vtx = &points.getVertices()[i];
			float tempDist = pyth2d(mX-vtx->x, mY-vtx->y);
			if(tempDist < minDist){
				minDist = tempDist;
				nearest = vtx;
			}
		}
		return nearest;
	}

	void drawPoints(vec3 color){
		points.updateGPU();
		points.draw(GL_POINTS, color);
	}
	Object getPoints() const {
		return points;
	}
	size_t getCount(){return pointCount; }
};

class Line{
	vec3 n, p;
	float param;
public:
	Line(vec3 a, vec3 b) {
		n = { a.y - b.y, b.x - a.x, 0 };
		p = a;
		param = dot(a, n);
		printf("Line: %fx + %fy = %f\n", n.x, n.y, param);
	}

	vec3 intersect(Line line) const {
		float f = line.n.x / n.x;
		float y = (f*param - line.param)/(f*n.y - line.n.y);
		float x = (param - n.y)/n.x;
		return vec3(x, y, 1);
	}

	bool through(const vec3& point) const {
		return std::abs(dot(point, n) - param) <= 0.01; 
	}
	vec3 getN() { return n; }
	vec3 getP() { return p; }
	float getParam() { return param; }
};

class LineCollection {
	Object lines;
	std::vector<Line> lineData;
public:
	void create() { lines.create(); }
	vec3 getBorderPoint(Line line, vec3 p, vec3 v, int dir) {
		while(insideBoundary(p.x, 1.05f, -1.05f) && insideBoundary(p.y, 1.05f, -1.05f)) {
			p = p + v*dir;
		}
		p.x = clamp(p.x, 1.1f, -1.1f);
		p.y = clamp(p.y, 1.1f, -1.1f);
		if(!insideBoundary(p.x, 1.05, -1.05)) {
			p.y = (line.getParam() - line.getN().x*p.x)/line.getN().y;
		}
		else if(!insideBoundary(p.y, 1.05f, -1.05f)) {
			p.x = (line.getParam() - line.getN().y*p.y)/line.getN().x;
		}
		return p;
	}
	void addLine(Line line) {
		lineData.push_back(line);
		vec3 v = { line.getN().y, -line.getN().x, 0 };
		vec3 a = getBorderPoint( line, line.getP(), v, 1 );
		vec3 b = getBorderPoint( line, line.getP(), v, -1 );

		lines.getVertices().push_back(a);
		lines.getVertices().push_back(b);
		
	}
	void drawLines(vec3 color) {
		if(lines.getVertices().size() < 2) return;
		lines.updateGPU();
		lines.draw(GL_LINES, color);
	}
};

PointCollection pointCollection;
LineCollection lineCollection;

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);

	glPointSize(10);
	glLineWidth(5);
	pointCollection.create();
	lineCollection.create();
	// create program for the GPU
	gpuProgram.create(vertexSource, fragmentSource, "outColor");
}


// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0.3, 0.3, 0.3, 0);     // background color
	glClear(GL_COLOR_BUFFER_BIT); // clear frame buffer

	lineCollection.drawLines(vec3(0, 1, 1));
	pointCollection.drawPoints(vec3(1, 0, 0));
	glutSwapBuffers(); // exchange buffers for double buffering
}

enum programModes {
	createPoints = 0,
	createLine = 1,
	moveLine = 2
};
int mode = -1;
// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	switch(key){
	case 'p': mode = createPoints; break;
	case 'l': mode = createLine;
	}
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {

}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {	// pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;
	//printf("Mouse moved to (%3.2f, %3.2f)\n", cX, cY);
}


void onMouse(int button, int state, int pX, int pY) {
	float cX = 2.0f * pX / windowWidth - 1;
	float cY = 1.0f - 2.0f * pY / windowHeight;
	float pointClickThreshold = 0.02;
	float lineClickThreshold = 0.01;

	if(button == GLUT_LEFT_BUTTON && state == GLUT_DOWN)
		switch(mode){
		case createPoints: pointCollection.addPoint(vec3(cX, cY, 1)); break;
		case createLine: {
			const static vec3* a,* b;
			static bool aSel = false;
			static bool bSel = false;

			const vec3* pt = pointCollection.findNearest(cX, cY);
			float dist = pyth2d(pt->x - cX, pt->y - cY);
			if(!aSel && dist < pointClickThreshold) {
				a = pt;
				aSel = true;
				if(aSel) printf("A point selected\n");
			}else if(!bSel && dist < pointClickThreshold) {
				b = pt;
				if(b != a) bSel = true;
				if(bSel) printf("B point selected\n");
			}	
			if(aSel && bSel){
				lineCollection.addLine(Line(*a, *b));
				aSel = bSel = false;
			}
		}
		}
		
	

}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
	glutPostRedisplay();
}
