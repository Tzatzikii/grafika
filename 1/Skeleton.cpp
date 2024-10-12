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
const char * const vertexSource = R"(
	#version 330
	layout(location = 0) in vec3 vertexPosition;
	void main() {
		gl_Position = vec4(vertexPosition, 1); // in NDC
	}
)";

const char * const fragmentSource = R"(
	#version 330
	uniform vec3 color;
	out vec4 fragmentColor;
	void main() {
		fragmentColor = vec4(color, 1);
	}
)";

GPUProgram gpuProgram;

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
		printf("Point %.1f, %.1f added\n", pos.x, pos.y);
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
		printf("Line added\n\tImplicit: %.1f x + %.1f y = %.1f\n\tParametric: r(t) = (%.1f, %.1f) + (%.1f, %.1f)t\n",
			n.x, n.y, param, a.x, a.y, n.y, -n.x);
	}

	float distanceFromPoint(float cX, float cY) const {
		return std::abs(n.x*cX + n.y*cY - param)/ pyth2d(n.x, n.y);
	}
	
	bool parallel(const Line& line) const {
		return std::abs(dot(n, vec3(line.getN().y, -line.getN().x))) < 0.01;
	}
	vec3 intersect(const Line& line) const {
		vec3 v1 = { n.x, n.y, -param };
		vec3 v2 = { line.n.x, line.n.y, -line.param };
		vec3 c = cross(v1, v2);
		return c/c.z;

	}	

	bool through(const vec3& point) const {
		return std::abs(dot(point, n) - param) <= 0.01; 
	}
	vec3 getN() const { return n; }
	vec3 getP() const { return p; }
	float getParam() { return param; }
	void setParam(float newParam) { param = newParam; }
};

class LineCollection {
	Object lines;
	std::vector<Line> lineData;
public:
	int selectedLine;
	void create() { 
		lines.create(); 
		selectedLine = -1;
	}
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

	int findNearest(float cX, float cY) {
		Line nearest = lineData[0];
		float minDist = nearest.distanceFromPoint(cX, cY);
		int index = 0;
		for(int i = 1; i < lineData.size(); i++) {
			Line temp = lineData[i];
			float dist = temp.distanceFromPoint(cX, cY);
			if(dist < minDist) {
				minDist = dist;
				nearest = temp;
				index = i;
			}
		}
		return index;
	}
	int distanceByIndex( int index, float cX, float cY ) {
		return lineData[index].distanceFromPoint(cX, cY);
	}

	bool parallel(int index1, int index2) const {
		return lineData[index1].parallel(lineData[index2]);
	}
	vec3 intersect(int index1, int index2) {
		return lineData[index1].intersect(lineData[index2]);
	}
	void updateLine(int index, float cX, float cY){
		vec3& v0 = lines.getVertices()[index*2];
		vec3& v1 = lines.getVertices()[index*2 + 1];
		Line line = lineData[index];
		v0 = getBorderPoint(line, vec3(cX, cY, 1), vec3(line.getN().y, -line.getN().x), 1);
		v1 = getBorderPoint(line, vec3(cX, cY, 1), vec3(line.getN().y, -line.getN().x), -1);
	}
	void moveTo(float cX, float cY) {
		Line& current = lineData[selectedLine];
		current.setParam(dot(current.getN(), vec3(cX, cY, 1)));
		updateLine(selectedLine, cX, cY);
	}
};

PointCollection pointCollection;
LineCollection lineCollection;

void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);

	glPointSize(10);
	glLineWidth(5);
	pointCollection.create();
	lineCollection.create();
	gpuProgram.create(vertexSource, fragmentSource, "outColor");
}


void onDisplay() {
	glClearColor(0.3, 0.3, 0.3, 0);   
	glClear(GL_COLOR_BUFFER_BIT); 

	lineCollection.drawLines(vec3(0, 1, 1));
	pointCollection.drawPoints(vec3(1, 0, 0));
	glutSwapBuffers(); 
}

enum programModes {
	createPoints = 0,
	createLine = 1,
	intersection = 2,
	moveLine = 3
};
int mode = -1;
// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	switch(key){
	case 'p': mode = createPoints; break;
	case 'l': mode = createLine; printf("Define lines\n"); break;
	case 'i': mode = intersection; printf("Intersect\n"); break;
	case 'm': mode = moveLine; printf("Move\n"); break;
	}
}

void onKeyboardUp(unsigned char key, int pX, int pY) {

}

void onMouseMotion(int pX, int pY) {
	float cX = 2.0f * pX / windowWidth - 1;
	float cY = 1.0f - 2.0f * pY / windowHeight;
	if(lineCollection.selectedLine > -1) {
		lineCollection.moveTo(cX, cY);
	}
}


void onMouse(int button, int state, int pX, int pY) {
	float cX = 2.0f * pX / windowWidth - 1;
	float cY = 1.0f - 2.0f * pY / windowHeight;
	float pointClickThreshold = 0.02;
	float lineClickThreshold = 0.01;

	if(button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {
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
			}else if(!bSel && dist < pointClickThreshold) {
				b = pt;
				if(b != a) bSel = true;
			}	
			if(aSel && bSel) {
				lineCollection.addLine(Line(*a, *b));
				aSel = bSel = false;
			}
		break;
		}
		case intersection: {
			static int e, f;
			static bool eSel = false;
			static bool fSel = false;

			int line = lineCollection.findNearest(cX, cY);
			float dist = lineCollection.distanceByIndex(line, cX, cY);
			if(!eSel && dist < lineClickThreshold) {
				e = line;
				eSel = true;
			}else if(!fSel && dist < lineClickThreshold) {
				f = line;
				if(f != e ) fSel = true;
			}
			if(eSel && fSel) {
				vec3 intersection = lineCollection.intersect(e, f);
				if(!lineCollection.parallel(e, f) && insideBoundary(intersection.x, 1, -1) && insideBoundary(intersection.y, 1, -1)) pointCollection.addPoint(intersection);
				eSel = fSel = false;
			}


		break;		
		}
		case moveLine: {
			int line = lineCollection.findNearest(cX, cY);
			float dist = lineCollection.distanceByIndex(line, cX, cY);
			if(dist < lineClickThreshold) lineCollection.selectedLine = line;
		break;
		}
		}
	
	}
	if(button == GLUT_LEFT_BUTTON && state == GLUT_UP) {
		switch(mode) {
		case moveLine: lineCollection.selectedLine = -1; break;
		}
	}

}

void onIdle() {
	glutPostRedisplay();
}
