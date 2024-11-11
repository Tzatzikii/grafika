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

inline float length(vec4 v){ return std::sqrt(dot(v,v)); }
inline vec4 normalize(vec4 v){ return v/length(v); }
inline vec3 vecPow(vec3 v, float pow){
	return {std::pow(v.x, pow), std::pow(v.y, pow), std::pow(v.z, pow)};
}
inline vec3 operator/(vec3 v0, vec3 v1) { return vec3(v0.x/v1.x, v0.y/v1.y, v0.z/v1.z);}
// vertex shader in GLSL: It is a Raw string (C++11) since it contains new line characters
const char * const vertexSource = R"(
	#version 330


	layout(location = 0) in vec2 vertexPos;
	out vec2 texCoord;

	void main() {
		texCoord = (vertexPos + vec2(1, 1))/2;
		gl_Position = vec4(vertexPos.x, vertexPos.y, 0, 1);
	}
)";

// fragment shader in GLSL
const char * const fragmentSource = R"(
	#version 330
	uniform sampler2D textureUnit;
	in vec2 texCoord;
	out vec4 outColor;

	void main() {
		outColor = texture(textureUnit, texCoord);
		//outColor = vec4(1,1,1,1);
	}
)";
const int WINDOW_WIDTH = 600, WINDOW_HEIGHT = 600;
const float EPSILON = 0.0001f;
GPUProgram gpuProgram;
struct Material {
	enum types{
		rough = 0,
		reflective = 1,
		refractive = 2
	};
	types type;
	vec3 ka, kd, ks, kappa, nu;
	float  shininess;
	Material(vec3 _kd, vec3 _ks, float _shininess) : ka(_kd * 3), kd(_kd), ks(_ks), shininess(_shininess) {
		type = rough; 
	}
	Material(vec3 _nu, vec3 _kappa, types _type) : nu(_nu), kappa(_kappa), type(_type){}
};
struct Ray{
	vec3 start, dir;
	bool out;
	Ray(vec3 _start, vec3 _dir, bool _out = true) : start(_start), dir(_dir), out(_out){
		dir = normalize(dir);
	}
};

class Camera {
	vec3 eye, lookat, right, up, vup;
	float fov;
	void build(){
		vec3 w = eye - lookat;
		float focus = length(w);
		right = normalize(cross(vup, w)) * focus * tanf(fov / 2);
		up = normalize(cross(w, right)) * focus * tanf(fov / 2);
	}
public:
	Camera(vec3 _eye, vec3 _lookat, vec3 _vup, float _fov) : eye(_eye), lookat(_lookat), vup(_vup), fov(_fov)  {
		build();
	}
	
	Ray getRay(int X, int Y) {
		vec3 dir = lookat + right*(2.0f*(X + 0.5f)/windowWidth - 1) + up*(2.0f*(Y + 0.5f)/windowHeight - 1) - eye;
		return Ray(eye, normalize(dir));
	}
	void rotate(float alpha){
		vec4 v = {eye.x, eye.y, eye.z, 0};

		v=v*RotationMatrix(alpha, {0, 1, 0});
		eye = {v.x, v.y, v.z};
		build();
	}
};


struct Hit{
	float t;
	vec3 pos;
	vec3 n;
	Material* material;
	Hit() { t = -1; }
};
struct Intersectable{
	Material* material;
	virtual Hit intersect(const Ray& ray) = 0;
};


class Cylinder : public Intersectable{
	vec3 base, top, dir;
	float r, h;
public:
	Cylinder(vec3 _base, vec3 _dir, float _r, float _h, Material* _material) : base(_base), dir(normalize(_dir)), r(_r), h(_h){
		material = _material;
		dir = normalize(dir);
		top = base+dir*h;
	}
	Hit intersect(const Ray& ray) {
		Hit hit;
		float a = dot(ray.dir, ray.dir)-dot(ray.dir, dir)*dot(ray.dir, dir);
		float b = 2*dot(ray.start, ray.dir)-2*dot(base, ray.dir) - 2*dot(ray.start, dir)*dot(ray.dir, dir) + 2*dot(base, dir)*dot(ray.dir, dir);
		float c = dot(ray.start - base, ray.start - base) - (dot(ray.start, dir)-dot(base, dir))*(dot(ray.start, dir)-dot(base, dir))  - r*r;
		float discr = b*b - 4*a*c;
		if(discr < 0) return hit;
		else discr = std::sqrt(discr);
		float t1 = (-b + discr)/(2*a), t2 = (-b - discr)/(2*a);
		if(t1 <= 0) return hit;
		hit.t = (t2 < t1) ? t2 : t1;
		hit.pos = ray.start + ray.dir*hit.t;
		if(!(dot((hit.pos - base), dir) >= 0 && dot((hit.pos - base), dir) <= h)) hit.t = -1;
		hit.n = 2*(hit.pos - base) - 2*(dot((hit.pos - base), dir))* dir;
		hit.n = normalize(hit.n);
		hit.material = material;
		return hit;
	}
};
class Cone : public Intersectable{
	vec3 base, top, dir;
	float alpha, h;
public:
	Cone(vec3 _top, vec3 _dir, float _alpha, float _h, Material* _material) : top(_top), dir(normalize(_dir)), alpha(_alpha), h(_h){
		material = _material;
		dir = normalize(dir);
		base = top-dir*h;
	} 
	Hit intersect(const Ray& ray){
		Hit hit;
		float a = dot(ray.dir, dir)*dot(ray.dir, dir) - dot(ray.dir, ray.dir)*std::cos(alpha)*std::cos(alpha);
		float b = 2*dot(ray.dir, dir)*(dot(dir, ray.start)-dot(dir, top)) - 2*dot(ray.dir, ray.start - top)*std::cos(alpha)*std::cos(alpha);
		float c = (dot(dir, ray.start)-dot(dir, top))*(dot(dir, ray.start)-dot(dir, top)) - dot(ray.start - top, ray.start - top)*std::cos(alpha)*std::cos(alpha);
		float discr = b*b - 4*a*c;
		if(discr < 0) return hit;
		else discr = std::sqrt(discr);
		float t1 = (-b + discr)/(2*a), t2 = (-b - discr)/(2*a);
		if(t1 <= 0) return hit;
		hit.t = (t2 < t1) ? t2 : t1;
		hit.pos = ray.start + ray.dir*hit.t;
		if(!(dot((hit.pos - top), dir) >= 0 && dot((hit.pos - top), dir) <= h)) hit.t = -1;
		hit.n = 2*(dot((hit.pos - top), dir))*dir - 2*(hit.pos - top)*std::cos(alpha)*std::cos(alpha);
		hit.n = normalize(hit.n);
		hit.material = material;
		return hit;

	}
	~Cone(){
		delete material;
	}
};

class Plane : public Intersectable{
	float y;
	int size, tileSize;
	Material* xorTrue,* xorFalse;
public:
	Plane(float _y, int _size, int _tileSize, Material* _xorTrue, Material* _xorFalse) : y(_y), size(_size), tileSize(_tileSize), xorTrue(_xorTrue), xorFalse(_xorFalse){} 
	Hit intersect(const Ray& ray){
		Hit hit;
		if(ray.dir.y == 0) return hit;
		float t = (y - ray.start.y)/ray.dir.y;
		if(t <= 0) return hit;
		hit.t = t;
		hit.pos = ray.start + ray.dir*hit.t;
		if(std::abs(hit.pos.x) > size/2 || std::abs(hit.pos.z) > size/2) hit.t = -1;
		if( ((int)floorf(hit.pos.x + size))%2 ^ ((int)floorf(hit.pos.z + size))%2 ) hit.material = xorTrue;
		else hit.material = xorFalse;
		hit.n = {0,1,0};
		return hit;

	}
};

struct Light{
	vec3 Le, dir;
	float in;
	Light(vec3 _Le, vec3 _dir, float _in) : Le(_Le*_in), dir(_dir), in(_in){
		dir = normalize(dir);
	};
};

Camera camera({0,1,4}, {0,0,0}, {0,1,0}, M_PI_4);

class Scene{
	std::vector<Intersectable*> objects;
	std::vector<Light*> lights;
	vec3 La;
public:
	vec4 image[WINDOW_WIDTH*WINDOW_HEIGHT];
	Scene(){
		La = {0.4f, 0.4f, 0.4f};
		Material* material1 = new Material({0.1, 0.2, 0.3}, {2, 2, 2}, 100);
		Material* material2 = new Material({0.3, 0, 0.2}, {2, 2, 2}, 20);
		Material* mirrorGolden = new Material({0.17, 0.35, 1.5}, {3.1, 2.7, 1.9}, Material::reflective);
		Material* water = new Material({1.3, 1.3, 1.3}, {0, 0, 0}, Material::refractive);
		Material* plasticOrange = new Material({0.3, 0.2, 0.1}, {2, 2, 2}, 50);
		Material* matteWhite = new Material({0.3f, 0.3f, 0.3f}, {0, 0, 0}, 0);
		Material* matteBlue = new Material({0.0f, 0.1f, 0.3f}, {0, 0, 0}, 0);
		objects.push_back(new Cone({0, 1, 0}, {-0.1, -1, -0.05}, 0.2, 2, material1));
		objects.push_back(new Cone({0, 1, 0.8}, {0.2, -1, 0}, 0.2, 2, material2));
		objects.push_back(new Plane(-1, 20, 1, matteBlue, matteWhite));
		objects.push_back(new Cylinder({1, -1, 0}, {0.1,1,0}, 0.3, 2, mirrorGolden));
		objects.push_back(new Cylinder({0, -1, -0.8}, {-0.2, 1, -0.1}, 0.3, 2, water));
		objects.push_back(new Cylinder({-1, -1, 0}, {0, 1, 0.1}, 0.3, 2, plasticOrange));

		lights.push_back(new Light({1,1,1},{1,1,1},2));

	}
	Hit firstIntersect(Ray ray) {
		Hit bestHit;
		for (Intersectable * object : objects) {
			Hit hit = object->intersect(ray); //  hit.t < 0 if no intersection
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  bestHit = hit;
		}
		if (dot(ray.dir, bestHit.n) > 0) bestHit.n = -bestHit.n;
		return bestHit;
	}
	bool shadowIntersect(Ray ray) {
		for (Intersectable * object : objects) if (object->intersect(ray).t > 0) return true;
		return false;
	}
	vec3 roughRad(Ray ray, Hit hit){
		vec3 outRadiance = hit.material->ka * La;
		for(Light* light : lights){
			Ray shadowRay(hit.pos + hit.n*EPSILON, light->dir);
			float cosTheta = dot(hit.n, light->dir);
			if(cosTheta > 0 && !shadowIntersect(shadowRay)){
				outRadiance = outRadiance + light->Le * hit.material->kd * cosTheta;
				vec3 halfway = normalize(-ray.dir + light->dir);
				float cosDelta = dot(hit.n, halfway);
				if(cosDelta > 0) {
					outRadiance = outRadiance + light->Le*hit.material->ks*std::pow(cosDelta, hit.material->shininess);
				}
				
			}
		}
		return outRadiance;
	}
	vec3 fresnel(vec3 dir, vec3 n, vec3 kappa, vec3 nu){
		float cosAlpha = -dot(dir, n);
		vec3 ones = {1, 1, 1};
		vec3 F0 = ((nu-ones)*(nu-ones)+kappa*kappa)/
			  ((nu+ones)*(nu+ones)+kappa*kappa);
		return F0 + (ones - F0) * std::pow(1 - cosAlpha, 5);
	}
	vec3 reflect(vec3 dir, vec3 n){
		return dir - n*dot(n, dir)*2;
	}

	vec3 refract(vec3 dir, vec3 n, float ns){
		float cosAlpha = -dot(dir, n);
		float disc = 1 - (1 - cosAlpha*cosAlpha)/ns/ns;
		if(disc < 0) return vec3(0,0,0);
		return dir/ns + n*(cosAlpha/ns - std::sqrt(disc));
	}
	vec3 reflectiveRad(Ray ray, Hit hit, vec3 outRadiance){
		Ray reflected(hit.pos + hit.n*EPSILON, reflect(ray.dir, hit.n), ray.out);
		outRadiance = outRadiance + trace(reflected)*(vec3(1,1,1) - fresnel(ray.dir, hit.n, hit.material->kappa, hit.material->nu));
		return outRadiance;
	}
	vec3 refractiveRad(Ray ray, Hit hit, vec3 outRadiance){
		float ior = ray.out ? hit.n.x : 1/hit.n.x;
		vec3 refractedDir = refract(ray.dir, hit.n, ior);
		if(length(refractedDir) > 0){
			Ray refractedRay = {hit.pos - hit.n*EPSILON, refractedDir, !ray.out};
			outRadiance = outRadiance + trace(refractedRay)*(vec3(1,1,1) - fresnel(ray.dir, hit.n, hit.material->kappa, hit.material->nu));
		}
		return outRadiance;
	}
	vec3 trace(Ray ray, int depth = 0){
		Hit hit = firstIntersect(ray);
		if(hit.t < 0) return La;
		vec3 outRadiance = {0,0,0};
		switch(hit.material->type){
			case Material::rough: outRadiance = roughRad(ray, hit); break;
			case Material::reflective: outRadiance = outRadiance + reflectiveRad(ray, hit, outRadiance); break;
			case Material::refractive: outRadiance = outRadiance + refractiveRad(ray, hit, outRadiance); break;
		}
		return outRadiance;	
	}
	void render(){
		for(int x = 0; x < WINDOW_WIDTH; x++){
			for(int y = 0; y < WINDOW_HEIGHT; y++){
				vec3 color = trace(camera.getRay(y, x));
				image[x*WINDOW_WIDTH + y] = {color.x, color.y, color.z,1};
			}
		}
	}

	~Scene(){
		for(Intersectable* o : objects) delete o;
	}

};
Scene scene;
class Screen{
	unsigned int vao, vbo, textureId;
	vec4* image;
public:
	void create(){
		image = scene.image;
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);
		glGenBuffers(1, &vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		float vertices[] = {-1, -1, -1, 1, 1, 1, 1, -1};
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);

		glGenTextures(1, &textureId);
		glBindTexture(GL_TEXTURE_2D, textureId);

		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, WINDOW_WIDTH, WINDOW_HEIGHT, 0, GL_RGBA, GL_FLOAT, image);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	}
	void draw(){
		glBindVertexArray(vao);
		int location = glGetUniformLocation(gpuProgram.getId(), "textureUnit");
		glUniform1i(location, 0);
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, textureId);
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
	}
};
Screen screen;
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	scene.render();
	screen.create();

	gpuProgram.create(vertexSource, fragmentSource, "outColor");
}


void onDisplay() {
	glClearColor(0, 0, 0, 0);   
	glClear(GL_COLOR_BUFFER_BIT); 
	screen.draw();

	glutSwapBuffers(); 
}

void onKeyboard(unsigned char key, int pX, int pY) {
	if(key == 'a') camera.rotate(M_PI_4);
	scene.render();
	screen.create();
	glutPostRedisplay();
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
	long time = glutGet(GLUT_ELAPSED_TIME);
}
