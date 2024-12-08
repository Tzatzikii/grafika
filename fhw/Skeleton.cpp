//=============================================================================================
// Mintaprogram: Zďż˝ld hďż˝romszďż˝g. Ervenyes 2019. osztol.
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
#include <time.h>
#include <random>
#include <chrono>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <cstring>

const int tessellationLevel = 6;

struct Camera { 
	vec3 wEye, wLookat, wVup, wDir;
	float fov, asp, fp, bp;	
public:
	Camera() {
		make();
	} 
	void make() {
		asp = (float)windowWidth / windowHeight;
		fov = 90.0f * (float)M_PI / 180.0f;
		fp = 0.1; bp = 100;
                wDir = wLookat - wEye;
	}
	void classicRotate(float angle){
		vec4 wEye4 = {wEye.x, wEye.y, wEye.z, 1};
		wEye4 = wEye4*RotationMatrix(angle, {0, 1, 0});
		wEye = vec3(wEye4.x, wEye4.y, wEye4.z);
		wDir = wLookat - wEye;
	}
	void advancedRotate(float angle){
		vec4 wDir4 = {wDir.x, wDir.y, wDir.z, 1};
		wDir4 = wDir4 * RotationMatrix(angle, {0, 1, 0});
		wDir = {wDir4.x, wDir4.y, wDir4.z};
	}
        void upDown(float angle){
                vec4 wDir4 = {wDir.x, wDir.y, wDir.z, 1};
                wDir4 = wDir4 * RotationMatrix(angle, cross(wDir, wVup));
                wDir = {wDir4.x, wDir4.y, wDir4.z};
        }
	void move(float units){
                vec3 wXY = normalize(vec3(wDir.x, 0, wDir.z));
		wEye = wEye + wXY * units;
	}
        void side(float units){
                vec3 wSidev = normalize(cross(normalize(wDir), wVup));
                wEye = wEye + wSidev * units;
        }
	mat4 V() { // view matrix: translates the center to the origin
		vec3 w = normalize(-wDir);
		vec3 u = normalize(cross(wVup, w));
		vec3 v = cross(w, u);
		return TranslateMatrix(-wEye) * mat4(u.x, v.x, w.x, 0,
			                                       u.y, v.y, w.y, 0,
			                                       u.z, v.z, w.z, 0,
			                                       0,   0,   0,   1);
	}

	mat4 P() { // projection matrix
		return mat4(1 / (tan(fov / 2)*asp), 0, 0, 0,
			        0, 1/tan(fov / 2), 0, 0,
			        0, 0, -(fp + bp)/(bp - fp), -1,
			        0, 0, -2 * fp*bp/(bp - fp),  0);
	}
};

struct Material {
	vec3 kd, ks, ka;
	float shininess;
};

struct Light {
	vec3 La, Le;
	vec4 wLightPos;
};
struct Triangle{
	vec4 r1, r2, r3;
};

struct VertexData {
	vec3 position, normal;
        vec2 texcoord;
};

struct RenderState {
	mat4	           MVP, M, Minv, V, P;
	std::vector<Light> lights;
	std::vector<Triangle> triangles;
	std::vector<VertexData> vertices;
	Texture * texture;
	vec3	           wEye, wDir;
};

class Shader : public GPUProgram {
public:
	virtual void Bind(RenderState state) = 0;

	void setUniformMaterial(const Material& material, const std::string& name) {
		setUniform(material.kd, name + ".kd");
		setUniform(material.ks, name + ".ks");
		setUniform(material.shininess, name + ".shininess");
	}

	void setUniformLight(const Light& light, const std::string& name) {
		setUniform(light.La, name + ".La");
		setUniform(light.Le, name + ".Le");
		setUniform(light.wLightPos, name + ".wLightPos");
	}
	void setUniformTriangle(const Triangle& triangle, const std::string& name) {
		setUniform(triangle.r1, name + ".r1");
		setUniform(triangle.r2, name + ".r2");
		setUniform(triangle.r3, name + ".r3");
	}
	void setUniformVertex(const VertexData& vertexData, const std::string& name) {
		setUniform(vertexData.position, name + ".position");
		setUniform(vertexData.normal, name + ".normal");
		//setUniform(vertexData.r3, name + ".r3");
	}
};


class DefaultShader : public Shader {
	const char * vertexSource = R"(
		#version 330
		precision highp float;


		layout(location = 0) in vec3  vtxPos;
		layout(location = 1) in vec3  vtxNorm;    

                uniform mat4 MVP;
                uniform mat4 Minv;
                uniform mat4 M;

                out vec3 wNorm;
                out vec3 wPos;

		void main() {
			gl_Position = vec4(vtxPos, 1) * MVP;
                        wNorm = (Minv * vec4(vtxNorm, 0)).xyz;
                        wPos = (vec4(vtxPos, 1) * M).xyz;
		}
	)";

	// fragment shader in GLSL
	const char * fragmentSource = R"(
		#version 330
		precision highp float;

                in vec3 wNorm;
                in vec3 wPos;

               // uniform vec3 wDir;
		
        	out vec4 fragmentColor;

		void main() {
                        vec3 lightPos = vec3(1, 1, 1);
                        vec3 L = normalize(vec3(1, 0, 0));
                        float d = length(lightPos - wPos);
                        float theta = acos(dot(wNorm, L));
                        fragmentColor = vec4(.4, .2, .25, 1) + vec4(.75, .2, .2, 1) * theta;
		}
	)";
public:
	DefaultShader() { create(vertexSource, fragmentSource, "fragmentColor"); }

	void Bind(RenderState state) {
		Use(); 		
		setUniform(state.MVP, "MVP");
		//setUniform(state.M, "M");
		setUniform(state.Minv, "Minv");
		//setUniform(state.wEye, "wEye");
                //setUniform(state.wDir, "wDir");
	}
};

class AxisShader : public Shader {

};
class PerlinShader : public Shader {
	const char * vertexSource = R"(
		#version 330

		layout(location = 0) in vec3 vtxPos;
		layout(location = 1) in vec3 vtxNorm;

		uniform mat4 MVP, M, Minv;

		out vec3 wNorm;
		out vec3 mNorm;
		out vec3 mPos;
		out vec3 wPos;

		void main(){
			mPos = vtxPos;
			mNorm = vtxNorm;
			wNorm = (Minv * vec4(mPos, 0)).xyz;
			wPos = (vec4(mPos, 1) * M).xyz;
			gl_Position = vec4(mPos, 1) * MVP;
		}
	)";
	const char * fragmentSource = R"(
		#version 330

		struct VertexData{
			vec3 position, normal;
		};

		//uniform int nVertices;
		//uniform VertexData vertices[4];

		in vec3 wNorm;
		in vec3 mNorm;
		in vec3 wPos;
		in vec3 mPos;
		
		
		out vec4 fragmentColor;

		void main(){
                       
                        fragmentColor =vec4(mNorm.xy, 0.1, 1);
		}
	)";
public:
PerlinShader() { create(vertexSource, fragmentSource, "fragmentColor"); }
	void Bind(RenderState state) {
		Use(); 		
		setUniform(state.MVP, "MVP");
		//setUniform(state.M, "M");
		//setUniform(state.Minv, "Minv");
		//setUniform(state.wEye, "wEye");
               // setUniform(state.wDir, "wDir");

	}
};
class TextureShader : public Shader {
        const char * vertexSource = R"(
                #version 330

                layout(location = 0) in vec3 vtxPos;
                layout(location = 2) in vec2 vtxUV;
                
                uniform mat4 MVP;

                out vec2 texcoord;

                void main(){
                        gl_Position = vec4(vtxPos, 1) * MVP;
                        texcoord = vtxUV;
                }
        )";
        const char * fragmentSource = R"(
                #version 330

                in vec2 texcoord;

                uniform sampler2D sampler;

                out vec4 fragmentColor;

                void main(){
                        fragmentColor = texture(sampler, texcoord);
                }
        
        )";
public:
        TextureShader() { create(vertexSource, fragmentSource, "fragmentColor"); }
        void Bind(RenderState state) {
                Use();
                setUniform(state.MVP, "MVP");
                setUniform(*state.texture, "sampler");
        }
};

class Geometry {
protected:
	unsigned int vao, vbo;        // vertex array object
public:
	std::vector<Triangle> triangles;
	Geometry() {
		glGenVertexArrays(1, &vao);
                glBindVertexArray(vao);
		glGenBuffers(1, &vbo); // Generate 1 vertex buffer object
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		
	}
	virtual void Draw() = 0;
	~Geometry() {
		glDeleteBuffers(1, &vbo);
		glDeleteVertexArrays(1, &vao);
	}
};
inline void readObjCoords(float& x, float& y, float& z){

}
class Mesh : public Geometry{
protected:
        std::vector<VertexData> vertices;
public:
        Mesh() {
                glEnableVertexAttribArray(0);
                glEnableVertexAttribArray(1);
                glEnableVertexAttribArray(2);
                glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, position));
                glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, normal));
                glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, texcoord));
        }

	Mesh(char * path) : Mesh() {
		std::ifstream is(path, std::ifstream::in);
		if(!is) { printf("Wrong path ! ! !\n"); return; }
		std::stringstream data;
		data << is.rdbuf();
		is.close();
		char line[251];
		std::vector<vec3> positions;
		std::vector<vec3> normals;
		std::vector<vec2> texcoords;
		printf("positions\n");
		data.getline(line, 250, '\n');
		while(line[1] != 'n'){
			std::stringstream ss(line);
			vec3 p;
			ss.seekg(1);
			ss >> std::skipws >> p.x >> p.y >> p.z;
			printf("%f %f %f\n", p.x, p.y, p.z);
			if(*line == 'v' && line[1] == ' ') positions.push_back(p);
			data.getline(line, 250, '\n');
			
		}
		printf("normals\n");
		while(line[1] != 't'){
			std::stringstream ss(line);
			vec3 n;
			ss.seekg(2);
			ss >> std::skipws >> n.x >> n.y >> n.z;
			printf("%f %f %f\n", n.x, n.y, n.z);
			if(*line == 'v' && line[1] == 'n') normals.push_back(n);
			data.getline(line, 250, '\n');
		}
		printf("texcoords\n");
		while(line[0] != 'f'){
			std::stringstream ss(line);
			vec2 t;
			ss.seekg(2);
			ss >> std::skipws >> t.x >> t.y;
			printf("%f %f\n", t.x, t.y);
			if(*line == 'v' && line[1] == 't') texcoords.push_back(t);
			data.getline(line, 250, '\n');
		}
		while(!data.eof()){
			std::stringstream ss(line);
			int idp, idt, idn;
			ss.seekg(2);
			do{
				((ss >> idp).ignore(1, '/') >> idt).ignore(1, '/') >> idn;
				printf("%d/%d/%d ", idp, idt, idn);
				if(*line == 'f') vertices.push_back({positions[idp-1], normals[idn-1], texcoords[idt-1]});
			}while(ss.peek() != EOF);
			printf("\n");
			data.getline(line, 250, '\n');
		}


	}
        void add(std::vector<VertexData> v){
                vertices.insert(vertices.end(), v.begin(), v.end());
        }
        void add(const VertexData& v){
                vertices.push_back(v);
        }
	std::vector<VertexData>& vertexArray() { return vertices;}
        virtual void Draw(){
                glBindVertexArray(vao);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
                glBufferData(GL_ARRAY_BUFFER, vertices.size()*sizeof(VertexData), &vertices[0], GL_STATIC_DRAW);
                glDrawArrays(GL_LINE_STRIP, 0, vertices.size());
        }

	

};
class Quad : public Mesh {
public:
        Quad(){
                vertices.push_back({{-1, 0, -1}, {0, 1, 0}, {0, 0}});
                vertices.push_back({{1, 0, -1}, {0, 1, 0}, {1, 0}});
                vertices.push_back({{-1, 0, 1}, {0, 1, 0}, {0, 1}});
                vertices.push_back({{1, 0, 1}, {0, 1, 0}, {1, 1}});
        }
};
class PerlinMesh : public Mesh {
	int nStrips;
	int nVtxPerStrip;
	int seed;
	const float distr = 0.06f;
public:  
	PerlinMesh(float t) : seed(time(NULL)) {
		std::random_device dev;
		std::default_random_engine dre(dev());
		nStrips = t;
		float x = 0;
		nVtxPerStrip = (t + 1) * 2;
		for(int i  = 0; i < t; i++){
			float y = 0;
			for(int j = 0; j <= t; j++){
				std::uniform_real_distribution<float> d(-x, y);
				vec3 normal1 = normalize({d(dre)+distr, d(dre)+distr, d(dre)+distr});
				vec3 normal2 = normalize({d(dre)+distr, d(dre)+distr, d(dre)+distr});
				if(i > 0){
					vertices.push_back(vertices[(i-1)*nVtxPerStrip + j*2+1]);
				}
				else{
					vertices.push_back({{i/t, d(dre), j/t}, normal1, {i/t, j/t}});
				}
				vertices.push_back({{(i+1)/t, d(dre), j/t}, normal2, {(i+1)/t, j/t}});
				y+=0.01;
			}
			x+=0.01;
		}
	}
	void Draw() override {
		glBindVertexArray(vao);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
                glBufferData(GL_ARRAY_BUFFER, vertices.size()*sizeof(VertexData), &vertices[0], GL_STATIC_DRAW);
		for(int i = 0; i < nStrips; i++){
			glDrawArrays(GL_TRIANGLE_STRIP, i * (nVtxPerStrip), nVtxPerStrip);
		}
	}
};
class Cube : public Mesh {
public:
        Cube(){
		// vertices.push_back({{-1, 0, -1}, {0, 1, 0}, {0, 0}});
                // vertices.push_back({{1, 0, -1}, {0, 1, 0}, {1, 0}});
                // vertices.push_back({{-1, 0, 1}, {0, 1, 0}, {0, 1}});
                // vertices.push_back({{1, 0, 1}, {0, 1, 0}, {1, 1}});
                //top quad
                vertices.push_back({{-1, 1, -1}, {0, 1, 0}, {.25, .3333f}});
                vertices.push_back({{1, 1, -1}, {0, 1, 0}, {.5, .3333f}});
                vertices.push_back({{-1, 1, 1}, {0, 1, 0}, {.25, 0}});
                vertices.push_back({{1, 1, 1}, {0, 1, 0}, {.5, 0}});

                //front face
                vertices.push_back({{-1, 1, 1}, {0, 0, 1}, {1, .3333f}});
                vertices.push_back({{1, 1, 1}, {0, 0, 1}, {.75, .3333f}});
                vertices.push_back({{-1, -1, 1}, {0, 0, 1}, {1, .6667}});
                vertices.push_back({{1, -1, 1}, {0, 0, 1}, {.75, .6667}});

                //right face
                vertices.push_back({{1, -1, 1}, {1, 0, 0}, {.75, .6667}});
                vertices.push_back({{1, 1, 1}, {1, 0, 0}, {.75, .3333f}});
                vertices.push_back({{1, -1, -1}, {1, 0, 0}, {.5, .6667}});
                vertices.push_back({{1, 1, -1}, {1, 0, 0}, {.5, .3333f}});

                //back face
                vertices.push_back({{1, 1, -1}, {0, 0, -1}, {.5, .3333f}});
                vertices.push_back({{1, -1, -1}, {0, 0, -1}, {.5, .6667}});
                vertices.push_back({{-1, 1, -1}, {0, 0, -1}, {.25, .3333f}});
                vertices.push_back({{-1, -1, -1}, {0, 0, -1}, {.25, .6667}});

                //left face
                vertices.push_back({{-1, 1, -1}, {-1, 0, 0}, {.25, .3333f}});
                vertices.push_back({{-1, -1, -1}, {-1, 0, 0}, {.25, .6667}});
                vertices.push_back({{-1, 1, 1}, {-1, 0, 0}, {0, .3333f}});
                vertices.push_back({{-1, -1, 1}, {-1, 0, 0}, {0, .6667}});

                //bottom face
                vertices.push_back({{-1, -1, 1}, {0, -1, 0}, {.25, 1}});
                vertices.push_back({{-1, -1, -1}, {0, -1, 0}, {.25, .6667}});
                vertices.push_back({{1, -1, 1}, {0, -1, 0}, {.5, 1}});
                vertices.push_back({{1, -1, -1}, {0, -1, 0}, {.5, .6667}});
        }
        
};

class Object {
	Shader *   shader;
        Mesh * mesh;
        Texture * texture;
protected:
	vec3 scale, translation, rotationAxis;
	float rotationAngle;
public:
	Object(Shader * _shader, Mesh * _mesh) :
		scale(1, 1, 1), translation(0, 0, 0), rotationAxis(0, 0, 1), rotationAngle(0) {
		shader = _shader;
                mesh = _mesh;
	}

	virtual void SetModelingTransform(mat4& M, mat4& Minv) {
		M = ScaleMatrix(scale) * RotationMatrix(rotationAngle, rotationAxis) * TranslateMatrix(translation);
		Minv = TranslateMatrix(-translation) * RotationMatrix(-rotationAngle, rotationAxis) * ScaleMatrix(vec3(1 / scale.x, 1 / scale.y, 1 / scale.z));
	}
        void translate(vec3 v) {
                translation = translation + v;
        }
        void setRotationAngle(float angle){
                rotationAngle += angle;
        }
        void setRotationAxis(vec3 axis){
                rotationAxis = axis;
        }
        void setTexture(Texture * _texture){
                texture = _texture;
        }
        void setScale(vec3 _scale){
                scale = _scale;
        }
        void setGeometry(Geometry * geometry) {

        }
	mat4 M(){
		return ScaleMatrix(scale) * RotationMatrix(rotationAngle, rotationAxis) * TranslateMatrix(translation);

	}

	void Draw(RenderState state) {
		
		mat4 M, Minv;
		SetModelingTransform(M, Minv);
		
		state.M = M;
		state.Minv = Minv;
		state.MVP = state.M * state.V * state.P;
                state.texture = texture;
		state.vertices = mesh->vertexArray();
                
		shader->Bind(state);
                mesh->Draw();
	}

	virtual void Animate(float tstart, float tend) { }
	~Object(){
		delete shader;
		delete mesh;
		delete texture;
	}
};
Camera camera;
class Scene {
	std::vector<Object *> objects;
	std::vector<VertexData> vertices;
	std::vector<Light> lights;
	Shader * phongShader;
	RenderState state;
public:
	void Build() {
                DefaultShader * shader = new DefaultShader();
                TextureShader * skyShader = new TextureShader();
		PerlinShader * perlinShader = new PerlinShader();

		Mesh * bucket = new Mesh("/home/kullancs/Documents/Modeling/Wavefront/vodor.obj");
                Mesh * cube = new Mesh("/home/kullancs/Documents/Modeling/Wavefront/cube.obj");
                Mesh * quad = new Quad();
		Mesh * perlin = new PerlinMesh(10);

                Texture * skyTexture = new Texture("./skybox50.bmp");

                struct Box : public Object {
                        float animatedRotation;
                        Box(Shader* _shader, Mesh* _mesh): Object(_shader, _mesh){}
                        void Animate(float Dt, float tend) override{
                                //rotationAngle += animatedRotation;
                        }
                };
                struct SkyBox : public Object {
                        SkyBox(Shader * _shader, Mesh * _mesh) : Object(_shader, _mesh){}
                        void Animate(float Dt, float tend) override{
                                translation = camera.wEye;
                        }
                };

               	Box * box = new Box(shader, bucket);
                box->setRotationAxis({0, 1, 0});
                box->setScale({10, 10, 10});
                //box->translate({-10, 15, 6});
                box->animatedRotation = 0.02f;

                Object * quadObject = new Object(shader, quad);
                quadObject->setScale({20, 20, 20});
                quadObject->translate({0, -2, 0});
		
		Object * perlinQuad = new Object(perlinShader, perlin);
		perlinQuad->translate({0, -2.0f, 0});
		perlinQuad->setScale({50, 50, 50});

                SkyBox * skyBox = new SkyBox(skyShader, cube);
                skyBox->setScale({50, 50, 50});
                skyBox->setTexture(skyTexture);


                objects.push_back(box);
                //objects.push_back(quadObject);
		//objects.push_back(perlinQuad);
               	objects.push_back(skyBox);

                camera.wEye = {0, 0, 4};
                camera.wLookat = {0, 1, 0};
                camera.wVup = {0, 1, 0};
                camera.make();
                
        }
	void Render() {
		
		state.wEye = camera.wEye;
		state.V = camera.V();
		state.P = camera.P();
		state.lights = lights;
		state.wDir = camera.wDir;
		for (Object * obj : objects){
			obj->Draw(state);
		}
	}

	void Animate(float tstart, float tend) {
		for (Object * obj : objects) obj->Animate(tstart, tend);
	}
};

Scene scene;

bool keyDown[sizeof(char)];
void onInitialization() {
	for(bool& key : keyDown){
		key = false;
	}
	glViewport(0, 0, windowWidth, windowHeight);
	glEnable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);
        glLineWidth(3);
	glPointSize(5);
        //glFrustum(-100, 100, -100, 100, 1, 100);
	scene.Build();
}

void onDisplay() {
	glClearColor(0.4f, 0.4f, 0.4f, 1.0f);							// background color 
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	scene.Render();
	glutSwapBuffers();									// exchange the two buffers
}

void onKeyboard(unsigned char key, int pX, int pY) { 
	keyDown[key] = true;
	
}

void onKeyboardUp(unsigned char key, int pX, int pY) { 
	keyDown[key] = false;
}

void onMouse(int button, int state, int pX, int pY) { }


void onMouseMotion(int pX, int pY) {
}

void onIdle() {
	static float tend = 0;
	const float dt = 0.1f;
	float tstart = tend;
	tend = glutGet(GLUT_ELAPSED_TIME) / 1000.0f;
        for(float t = tstart; t < tend; t+=dt){
                float Dt = fminf(dt, tend - t);
                scene.Animate(Dt, tend);
                glutPostRedisplay();
        }
	if(keyDown['w']){
		camera.move(0.05);
		glutPostRedisplay();
	}
	if(keyDown['s']){
		camera.move(-0.05);
		glutPostRedisplay();
	}
	if(keyDown['a']){
		camera.side(-0.05);
		glutPostRedisplay();
	}
	if(keyDown['d']){
		camera.side(0.05);
		glutPostRedisplay();
	}
	if(keyDown['j']){
		camera.advancedRotate(0.05);
		glutPostRedisplay();
	}
        if(keyDown['l']){
		camera.advancedRotate(-0.05);
		glutPostRedisplay();
	}
        if(keyDown['i']){
		camera.upDown(0.05);
		glutPostRedisplay();
	}
        if(keyDown['k']){
		camera.upDown(-0.05);
		glutPostRedisplay();
	}
        
}
