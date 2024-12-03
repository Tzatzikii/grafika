//=============================================================================================
// Computer Graphics Sample Program: 3D engine-let
// Shader: Gouraud, Phong, NPR
// Material: diffuse + Phong-Blinn
// Texture: CPU-procedural
// Geometry: sphere, tractricoid, torus, mobius, klein-bottle, boy, dini
// Camera: perspective
// Light: point or directional sources
//=============================================================================================
#include "framework.h"

//---------------------------
template<class T> struct Dnum { // Dual numbers for automatic derivation
//---------------------------
	float f; // function value
	T d;  // derivatives
	Dnum(float f0 = 0, T d0 = T(0)) { f = f0, d = d0; }
	Dnum operator+(Dnum r) { return Dnum(f + r.f, d + r.d); }
	Dnum operator-(Dnum r) { return Dnum(f - r.f, d - r.d); }
	Dnum operator*(Dnum r) {
		return Dnum(f * r.f, f * r.d + d * r.f);
	}
	Dnum operator/(Dnum r) {
		return Dnum(f / r.f, (r.f * d - r.d * f) / r.f / r.f);
	}
};

// Elementary functions prepared for the chain rule as well
template<class T> Dnum<T> Exp(Dnum<T> g) { return Dnum<T>(expf(g.f), expf(g.f)*g.d); }
template<class T> Dnum<T> Sin(Dnum<T> g) { return  Dnum<T>(sinf(g.f), cosf(g.f)*g.d); }
template<class T> Dnum<T> Cos(Dnum<T>  g) { return  Dnum<T>(cosf(g.f), -sinf(g.f)*g.d); }
template<class T> Dnum<T> Tan(Dnum<T>  g) { return Sin(g) / Cos(g); }
template<class T> Dnum<T> Sinh(Dnum<T> g) { return  Dnum<T>(sinh(g.f), cosh(g.f)*g.d); }
template<class T> Dnum<T> Cosh(Dnum<T> g) { return  Dnum<T>(cosh(g.f), sinh(g.f)*g.d); }
template<class T> Dnum<T> Tanh(Dnum<T> g) { return Sinh(g) / Cosh(g); }
template<class T> Dnum<T> Log(Dnum<T> g) { return  Dnum<T>(logf(g.f), g.d / g.f); }
template<class T> Dnum<T> Pow(Dnum<T> g, float n) {
	return  Dnum<T>(powf(g.f, n), n * powf(g.f, n - 1) * g.d);
}

typedef Dnum<vec2> Dnum2;

const int tessellationLevel = 6;

//---------------------------
struct Camera { // 3D camera
//---------------------------
	vec3 wEye, wLookat, wVup, wDir;   // extrinsic
	float fov, asp, fp, bp;		// intrinsic
public:
	Camera() {
		make();
	} 
	void make() {
		asp = (float)windowWidth / windowHeight;
		fov = 45.0f * (float)M_PI / 180.0f;
		fp = 1; bp = 20;
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
	void move(float units){
		wEye = wEye + wDir * units;
	}
	mat4 V() { // view matrix: translates the center to the origin
		vec3 w = normalize(-wDir);
		vec3 u = normalize(cross(wVup, w));
		vec3 v = cross(w, u);
		return TranslateMatrix(wEye * (-1)) * mat4(u.x, v.x, w.x, 0,
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

//---------------------------
struct Material {
//---------------------------
	vec3 kd, ks, ka;
	float shininess;
};

//---------------------------
struct Light {
//---------------------------
	vec3 La, Le;
	vec4 wLightPos; // homogeneous coordinates, can be at ideal point
};
struct Triangle{
		vec3 r1, r2, r3;
};

struct VertexData {
		vec3 position, normal;
	};
//---------------------------
class CheckerBoardTexture : public Texture {
//---------------------------
public:
	CheckerBoardTexture(const int width, const int height) : Texture() {
		std::vector<vec4> image(width * height);
		const vec4 white(0.3, 0.3, 0.3, 1), blue(0, 0.1, 0.3, 1);
		for (int x = 0; x < width; x++){ 
			for (int y = 0; y < height; y++) {
				image[y * width + x] = (x & 1) ^ (y & 1) ? blue : white;
			}
		}
		create(width, height, image, GL_NEAREST);
	}
};

//---------------------------
struct RenderState {
//---------------------------
	mat4	           MVP, M, Minv, V, P;
	Material *         material;
	std::vector<Light> lights;
	std::vector<Triangle> triangles;
	Texture *          texture;
	vec3	           wEye;
};

//---------------------------
class Shader : public GPUProgram {
//---------------------------
public:
	virtual void Bind(RenderState state) = 0;

	void setUniformMaterial(const Material& material, const std::string& name) {
		setUniform(material.kd, name + ".kd");
		setUniform(material.ks, name + ".ks");
		//setUniform(material.ka, name + ".ka");
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
};


//---------------------------
class PhongShader : public Shader {
//---------------------------
	const char * vertexSource = R"(
		#version 330
		precision highp float;

		struct Light {
			vec3 La, Le;
			vec4 wLightPos;
		};

		uniform mat4  MVP, M, Minv; // MVP, Model, Model-inverse
		uniform Light[8] lights;    // light sources 
		uniform int   nLights;
		uniform vec3  wEye;         // pos of eye

		layout(location = 0) in vec3  vtxPos;            // pos in modeling space
		layout(location = 1) in vec3  vtxNorm;      	 // normal in modeling space

		out vec3 wNormal;		    // normal in world space
		out vec3 wView;             // view in world space
		out vec3 wLight[8];		    // light dir in world space
		out vec2 texcoord;
		out vec4 wPos;


		void main() {
			gl_Position = vec4(vtxPos, 1) * MVP; // to NDC
			// vectors for radiance computation
			wPos = vec4(vtxPos, 1) * M;
			for(int i = 0; i < nLights; i++) {
				wLight[i] = lights[i].wLightPos.xyz; /* wPos.w - wPos.xyz * lights[i].wLightPos.w; */
			}

		    wView  = wEye * wPos.w - wPos.xyz;
		    wNormal = (Minv * vec4(vtxNorm, 0)).xyz;
		    texcoord.x = (vtxPos.x+1)/2;
		    texcoord.y = (vtxPos.z+1)/2;
		}
	)";

	// fragment shader in GLSL
	const char * fragmentSource = R"(
		#version 330
		precision highp float;

		struct Light {
			vec3 La, Le;
			vec4 wLightPos;
		};

		struct Material {
			vec3 kd, ks, ka;
			float shininess;
		};

		struct Triangle {
			vec3 r1, r2, r3;
		};
		uniform mat4 M, Minv;
		uniform Material material;
		uniform Light[8] lights;    // light sources 
		uniform int   nLights;
		uniform sampler2D paneTexture;
		uniform int pane;

		uniform Triangle[60] triangles; //5*12 triangle

		in vec3 wNormal;       // interpolated world sp normal
		in vec3 wView;         // interpolated world sp view
		in vec3 wLight[8];     // interpolated world sp illum dir
		in vec2 texcoord;
		in vec4 wPos;
		
		int shadowIntersect(vec3 start, vec3 dir){
			for(int i = 0; i < 60; i++){
				vec3 r1 = (vec4(triangles[i].r1, 1)* M).xyz;
				vec3 r2 = (vec4(triangles[i].r2, 1) * M).xyz;
				vec3 r3 = (vec4(triangles[i].r3, 1) * M).xyz;
				vec3 n = cross(r2 - r1, r3 - r1);
				float t = dot(r1 - start, n)/dot(dir, n);
				if(t < 0) return 0;
				vec3 p = start + dir*t;
				if(	dot(cross(r2 - r1, p - r1), n) > 0 &&
					dot(cross(r3 - r2, p - r2), n) > 0 &&
					dot(cross(r1 - r3, p - r3), n) > 0) return 1;
			}
			return 0;
		}
        	out vec4 fragmentColor; // output goes to frame buffer
		void main() {
			const float EPSILON = 0.0001f;
			vec3 N = normalize(wNormal);
			vec3 V = normalize(wView); 
			if (dot(N, V) < 0) N = -N;	// prepare for one-sided surfaces like Mobius or Klein
			vec3 ka = material.kd * 3;
			vec3 kd = material.kd;
			vec3 texColor = pane > 0 ? texture(paneTexture, texcoord).rgb : vec3(1, 1, 1);
			vec3 radiance = vec3(0, 0, 0);
			for(int i = 0; i < nLights; i++) {
				vec3 L = normalize(wLight[i]);
				vec3 H = normalize(L + V);
				float cost = dot(N,L);
				radiance += ka * lights[i].La ;
				if(cost > 0 &&  shadowIntersect(wPos.xyz + N*EPSILON, -L) < 1 ){
					float cosd = max(dot(N,H), 0);
					// kd and ka are modulated by the texture
					if(pane > 0){
						radiance += texColor;
					}
					else radiance += (kd * texColor * cost + material.ks * pow(cosd, material.shininess)) * lights[i].Le;
				}
			}
			if(pane > 0){
				radiance = texColor*2.5;
			}
			fragmentColor = vec4(radiance, 1);
		}
	)";
public:
	PhongShader() { create(vertexSource, fragmentSource, "fragmentColor"); }

	void Bind(RenderState state) {
		Use(); 		// make this program run
		setUniform(state.MVP, "MVP");
		setUniform(state.M, "M");
		setUniform(state.Minv, "Minv");
		setUniform(state.wEye, "wEye");
		setUniformMaterial(*state.material, "material");
		if(state.texture != nullptr){
			setUniform(1, "pane");
		}
		else{
			setUniform(0, "pane");
		}
		setUniform((int)state.lights.size(), "nLights");
		for (unsigned int i = 0; i < state.lights.size(); i++) {
			setUniformLight(state.lights[i], std::string("lights[") + std::to_string(i) + std::string("]"));
		}
		for (unsigned int i = 0; i < state.triangles.size(); i++) {
			setUniformTriangle(state.triangles[i], std::string("triangles[") + std::to_string(i) + std::string("]"));
		}
	}
};

//---------------------------
class Geometry {
//---------------------------
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

class ChessPane : public Geometry{
	int yLevel = -1;
	bool pane = true;
	Shader *   shader;
	Material * material;
	RenderState state;
public:
	ChessPane(RenderState _state, Shader* _shader, Material* _material)\
	: state(_state), shader(_shader), material(_material){}
	void Draw(){
		glBindVertexArray(vao);
		mat4 M, Minv;;
		state.M = ScaleMatrix(vec3(10, 0, 10)) * TranslateMatrix({0, -1, 0});
		state.Minv = TranslateMatrix({0, 1, 0})*ScaleMatrix(vec3(1/10, 0, 1/10));
		state.MVP = state.M * state.V * state.P;
		state.material = material;
		shader->Bind(state);

		VertexData data[] = {{{1, 0, 1},{0, 1, 0}}, {{1, 0, -1}, {0, 1, 0}}, {{-1, 0, -1}, {0, 1, 0}}, {{-1, 0, 1}, {0, 1, 0}}};
		glBufferData(GL_ARRAY_BUFFER, 4*sizeof(VertexData), data, GL_STATIC_DRAW);
		glEnableVertexAttribArray(0);
		glEnableVertexAttribArray(1);

		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, position));
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, normal));

		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
	}
};
inline vec4 vec3to4(vec3 v, float w){ return vec4(v.x, v.y, v.z, w); }
//---------------------------
class ParamSurface : public Geometry {
//---------------------------
	
	
	
	unsigned int nVtxPerStrip, nStrips, nTriangles;
	bool pane = false;
public:
	ParamSurface() { nVtxPerStrip = nStrips = 0; }

	virtual void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) = 0;

	VertexData GenVertexData(float u, float v) {
		VertexData vtxData;
		Dnum2 X, Y, Z;
		Dnum2 U(u, vec2(1, 0)), V(v, vec2(0, 1));
		eval(U, V, X, Y, Z);
		vtxData.position = vec3(X.f, Y.f, Z.f);
		vec3 drdU(X.d.x, Y.d.x, Z.d.x), drdV(X.d.y, Y.d.y, Z.d.y);
		vtxData.normal = cross(drdU, drdV);
		return vtxData;
	}

	void create(int N = tessellationLevel, int M = tessellationLevel) {
		nVtxPerStrip = (M + 1) * 2;
		N = 1;
		nStrips = N;
		nTriangles = M*2;
		std::vector<VertexData> vtxData;	// vertices on the CPU
		
		for (int i = 0; i < N; i++) {
			for (int j = 0; j <= M; j++) {
				vtxData.push_back(GenVertexData((float)j / M, (float)i / N));
				vtxData.push_back(GenVertexData((float)j / M, (float)(i + 1) / N));
			}
		}
		for(int i = 1; i < nVtxPerStrip * nStrips - 1; i++){
			triangles.push_back({vtxData[i-1].position, vtxData[i].position,
						vtxData[i+1].position});
		}
		glBufferData(GL_ARRAY_BUFFER, nVtxPerStrip * nStrips * sizeof(VertexData), &vtxData[0], GL_STATIC_DRAW);
		// Enable the vertex attribute arrays
		glEnableVertexAttribArray(0);  // attribute array 0 = POSITION
		glEnableVertexAttribArray(1);  // attribute array 1 = NORMAL
		// attribute array, components/attribute, component type, normalize?, stride, offset
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, position));
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, normal));
	}

	void Draw() {
		glBindVertexArray(vao);
		for (unsigned int i = 0; i < nStrips; i++) glDrawArrays(GL_TRIANGLE_STRIP, i *  nVtxPerStrip, nVtxPerStrip);
	}
};



//---------------------------
class Cylinder : public ParamSurface {
//---------------------------
public:
	Cylinder() { create(); }
	void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) {
		U = U * 2.0f * M_PI;
		V = V * 2;
		X = Cos(U)*0.3; 
		Z = Sin(U)*0.3; 
		Y = V;
	}
};
class Cone : public ParamSurface {
public:
        Cone() { create(); }
        void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z){
		U = U * M_PI * 2.0f;
		V = V * 2 * -1;
		X = Cos(U)*tanf(0.2) * V;
		Z = Sin(U)*tanf(0.2) * V;
		Y = V;
	}
};


//---------------------------
struct Object {
//---------------------------
	Shader *   shader;
	Material * material;
	Geometry * geometry;
	vec3 scale, translation, rotationAxis;
	float rotationAngle;
public:
	Object(Shader * _shader, Material * _material, Geometry * _geometry) :
		scale(vec3(1, 1, 1)), translation(vec3(0, 0, 0)), rotationAxis(0, 0, 1), rotationAngle(0) {
		shader = _shader;
		material = _material;
		geometry = _geometry;
	}

	virtual void SetModelingTransform(mat4& M, mat4& Minv) {
		M = ScaleMatrix(scale) * RotationMatrix(rotationAngle, rotationAxis) * TranslateMatrix(translation);
		Minv = TranslateMatrix(-translation) * RotationMatrix(-rotationAngle, rotationAxis) * ScaleMatrix(vec3(1 / scale.x, 1 / scale.y, 1 / scale.z));
	}

	void Draw(RenderState state) {
		
		mat4 M, Minv;
		SetModelingTransform(M, Minv);
		state.M = M;
		state.Minv = Minv;
		state.MVP = state.M * state.V * state.P;
		state.material = material;
		shader->Bind(state);
		geometry->Draw();
	}

	virtual void Animate(float tstart, float tend) { rotationAngle = 0.8f * tend; }
};

Camera camera; // 3D camera
//---------------------------
class Scene {
//---------------------------
	std::vector<Object *> objects;
	std::vector<Light> lights;\
	Shader * phongShader;
public:
	void Build() {
		// Shaders
		phongShader = new PhongShader();;

		// Materials
		Material * cyan = new Material;
		cyan->kd = vec3(0.1, 0.2, 0.3);
		cyan->ks = vec3(2, 2, 2);
		cyan->shininess = 100;

		Material * magenta = new Material;
		magenta->kd = vec3(0.3f, 0.0f, 0.2f);
		magenta->ks = vec3(2, 2, 2);
		magenta->shininess = 20;

		Material * orange = new Material;
		orange->kd = vec3(0.3f, 0.2f, 0.1f);
		orange->ks = vec3(2, 2, 2);
		orange->shininess = 50;

		Material * gold = new Material;
		gold->kd = vec3(0.17, 0.35f, 1.5f);
		gold->ks = vec3(3.1, 2.7, 1.9);
		gold->shininess = 50;

		Material * water = new Material;
		water->kd = vec3(1.3, 1.3, 1.3);
		water->ks = vec3(1, 1, 1);
		water->shininess = 50;

		// Geometries
		Geometry * cone1 = new Cone();
		Geometry * cone2 = new Cone();

		Object * cyanCone = new Object(phongShader, cyan, new Cone());
		vec3 cTop = vec3(0, 1, 0);
		vec3 cDir = normalize(vec3(-0.1, -1, -0.05));
		cyanCone->translation = cTop;
		cyanCone->rotationAngle = acosf(dot(normalize({0, -1, 0}), cDir));
		cyanCone->rotationAxis = normalize(cross({0, -1, 0}, cDir));
		objects.push_back(cyanCone);

		Object * magentaCone = new Object(phongShader, magenta, new Cone());
		vec3 mDir = normalize(vec3(0.2, -1, 0));
		magentaCone->translation = vec3(0, 1, 0.8);
		magentaCone->rotationAngle = acosf(dot(normalize({0, -1, 0}), mDir));
		magentaCone->rotationAxis = normalize(cross({0, -1, 0}, mDir));
		objects.push_back(magentaCone);

		Object * orangeCylinder = new Object(phongShader, orange, new Cylinder());
		vec3 oDir = normalize(vec3(0, 1, 0.1));
		orangeCylinder->translation = {-1, -1, 0};
		orangeCylinder->rotationAngle = acosf(dot({0, 1, 0}, oDir));
		orangeCylinder->rotationAxis = normalize(cross({0, 1, 0}, oDir));
		objects.push_back(orangeCylinder);

		Object * goldenCylinder = new Object(phongShader, gold, new Cylinder());
		vec3 gDir = normalize(vec3(0.1, 1, 0));
		goldenCylinder->translation = {1, -1, 0};
		goldenCylinder->rotationAngle = acosf(dot({0, 1, 0}, gDir));
		goldenCylinder->rotationAxis = normalize(cross({0, 1, 0}, gDir));
		objects.push_back(goldenCylinder);

		Object * waterCylinder = new Object(phongShader, water, new Cylinder());
		vec3 wDir = normalize(vec3(-0.2, 1, -0.1));
		waterCylinder->translation = {0, -1, -0.8};
		waterCylinder->rotationAngle = acosf(dot({0, 1, 0}, wDir));
		waterCylinder->rotationAxis = normalize(cross({0, 1, 0}, wDir));
		objects.push_back(waterCylinder);


		// Camera
		camera.wEye = vec3(0, 1, 4);
		camera.wLookat = vec3(0, 1, 0);
		camera.wVup = vec3(0, 1, 0);
		
		camera.wDir = camera.wLookat - camera.wEye;;

		// Lights
		Light light;
		light.wLightPos = vec4(1, 1, 1, 0);	// ideal point -> directional light source
		light.La = vec3(0.4f, 0.4f, 0.4f);
		light.Le = vec3(2, 2, 2);
		lights.push_back(light);
	}

	void Render() {
		RenderState state;
		state.wEye = camera.wEye;
		state.V = camera.V();
		state.P = camera.P();
		state.lights = lights;
		state.texture = nullptr;
		for (Object * obj : objects){
			state.triangles.insert(state.triangles.end(), obj->geometry->triangles.begin(), obj->geometry->triangles.end());
		}
		for (Object * obj : objects){
			obj->Draw(state);
		}
		state.texture = new CheckerBoardTexture(20, 20);
		Material * pane = new Material;
		pane->kd = vec3(0.4, 0.4, 0.4);
		pane->ks = vec3(1, 1, 1);
		pane->shininess = 100;
		ChessPane chessPane(state, phongShader, pane);
		chessPane.Draw();

	}

	void Animate(float tstart, float tend) {
		for (Object * obj : objects) obj->Animate(tstart, tend);
	}
};

Scene scene;

// Initialization, create an OpenGL context
bool keyDown[sizeof(char)];
void onInitialization() {
	for(bool& key : keyDown){
		key = false;
	}
	glViewport(0, 0, windowWidth, windowHeight);
	glEnable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);
	scene.Build();
}

// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0.4f, 0.4f, 0.4f, 1.0f);							// background color 
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear the screen
	scene.Render();
	glutSwapBuffers();									// exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) { 
	keyDown[key] = true;
	
}
// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) { 
	keyDown[key] = false;
}
// Mouse click event
void onMouse(int button, int state, int pX, int pY) { }

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	static float tend = 0;
	const float dt = 0.1f; // dt is ”infinitesimal”
	float tstart = tend;
	tend = glutGet(GLUT_ELAPSED_TIME) / 1000.0f;

	if(keyDown['w']){
		camera.move(0.1);
	}
	if(keyDown['s']){
		camera.move(-0.1);
	}
	if(keyDown['a']){
		camera.advancedRotate(0.05);
	}
	if(keyDown['d']){
		camera.advancedRotate(-0.05);
	}
	if(keyDown['r']){
		camera.classicRotate(0.05);
	}
	glutPostRedisplay();
}
