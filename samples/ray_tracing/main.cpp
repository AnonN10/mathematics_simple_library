#include <cstdlib>
#include <memory>
#include <chrono>

#include <SDL.h>

#include "thread_pool.h"
#include "mathematics_simple_library.hpp"

using namespace Maths;

using ivec2 = vec_static_t<2, int>;
using ivec3 = vec_static_t<3, int>;
using ivec4 = vec_static_t<4, int>;
using vec2 = vec_static_t<2, float>;
using vec3 = vec_static_t<3, float>;
using vec4 = vec_static_t<4, float>;
using mat3 = mat_static_t<3, 3, float>;
using mat4 = mat_static_t<4, 4, float>;

mat4 projection(float l, float r, float b, float t, float n, float f) {
    return {
        2*n/(r-l),	0,			(r+l)/(r-l),	0,
        0,			2*n/(t-b),	(t+b)/(t-b),	0,
        0,			0,			-(f+n)/(f-n),	-(2*f*n)/(f-n),
        0,			0,			-1,				0,
    };
}

mat4 projection(float fov, float aspectRatio, float near, float far) {
    float tangent = tan(fov/2);
    float top = near * tangent;
    float right = top * aspectRatio;
    return projection(-right, right, -top, top, near, far);
}

vec3 homogeneous_division(const vec4& in) {
    return vec3(in/in[3]);
}

template<ConceptVector A>
constexpr auto abs(const A& a) {
    return unary_operation(a, [](auto x){ using std::abs; return abs(x); });
}

template<ConceptVector A>
constexpr auto sign(const A& a) {
    return unary_operation(a, [](auto x){ return std::signbit(x) ? -1 : 1; });
}

template<ConceptVector A, typename B>
constexpr auto step(const A& a, const B& b) {
    return binary_operation(a, b, [](auto x, auto y){ return x < y ? 0 : 1; });
}

mat3 tangent_space_from_normal(const vec3& n) {
    float s = (n[2] < 0.0)?-1.0:1.0;
    float a = -1.0 / (s + n[2]);
    float b = n[0] * n[1] * a;
    vec3 t = vec3{1.0f + s * n[0]* n[0] * a, s * b, -s * n[0]};
    vec3 bt = vec3{b, s + n[1] * n[1] * a, -n[1]};
    return join(join(as_column(t), as_column(n)), as_column(bt));
}

float checker_pattern(const vec2& uv, float scale) {
    auto u_mod_1 = eucmod(scale * uv[0], 1.0f);
    auto v_mod_1 = eucmod(scale * uv[1], 1.0f);
    return ((u_mod_1 < 0.5 && v_mod_1 < 0.5) || (u_mod_1 > 0.5 && v_mod_1 > 0.5))? 1.0f : 0.0f;
}

float beer_lambert_law(float mu, float distance) {
    return std::exp(-mu*distance);
}

struct IntersectionResult {
    bool hit = false;
    float t = std::numeric_limits<float>::max();
    vec3 normal {0, 1, 0};
    vec2 uv {0, 0};
};

struct Shape {
    Shape() = default;
    virtual ~Shape() = default;

    virtual IntersectionResult Trace(const vec3& ray_origin, const vec3& ray_direction, float t_min, float t_max) = 0;
};

struct AABB : Shape {
    vec3 bmin, bmax;

    AABB() = default;
    AABB(const vec3& bmin, const vec3& bmax) : bmin(bmin), bmax(bmax) {} 

    IntersectionResult Trace(const vec3& ro, const vec3& rd, float t_min, float t_max) {
        IntersectionResult result;

        auto rcp_rd = 1/rd;
        auto t_bmin = (bmin - ro) * rcp_rd;
        auto t_bmax = (bmax - ro) * rcp_rd;
        auto t_1 = Maths::min(t_bmin, t_bmax);
        auto t_2 = Maths::max(t_bmin, t_bmax);
        float t_near = std::max(std::max(t_1[0], t_1[1]), t_1[2]);
        float t_far = std::min(std::min(t_2[0], t_2[1]), t_2[2]);

        bool outside = 0 <= t_near;
        float t = outside? t_near : t_far;
        if(t_near < t_far && t >= t_min && t <= t_max) {
            result.t = t;
            auto center = (bmin + bmax)*0.5f;
            auto point_in_aabb = ro + t*rd - center;
            auto aabb_half_extents = abs((bmax - bmin)*0.5f);
            auto point_in_box = point_in_aabb/aabb_half_extents;
            auto abs_point_in_box = abs(point_in_box);
            result.normal = vec3{
                (abs_point_in_box[0] >= abs_point_in_box[1] && abs_point_in_box[0] >= abs_point_in_box[2])?
                (std::signbit(point_in_aabb[0]) ? -1.0f : 1.0f):
                0.0f,
                (abs_point_in_box[1] > abs_point_in_box[2] && abs_point_in_box[1] > abs_point_in_box[0])?
                (std::signbit(point_in_aabb[1]) ? -1.0f : 1.0f):
                0.0f,
                (abs_point_in_box[2] > abs_point_in_box[0] && abs_point_in_box[2] > abs_point_in_box[1])?
                (std::signbit(point_in_aabb[2]) ? -1.0f : 1.0f):
                0.0f
            };
            if(!outside) result.normal = -result.normal;
            vec3 normal_space_point_in_box = transpose(tangent_space_from_normal(result.normal))*(point_in_box+1)*0.5;
            result.uv = vec_ref({normal_space_point_in_box[0], normal_space_point_in_box[2]});
            result.hit = true;
        }

        return result;
    }
};

struct Sphere : Shape {
    float radius = 1.0f;

    Sphere() = default;
    Sphere(float radius) : radius(radius) {}

    IntersectionResult Trace(const vec3& ro, const vec3& rd, float t_min, float t_max) {
        IntersectionResult result;
        //a quadratic equation
        float a = dot(rd, rd);
        float b = 2 * dot(ro, rd);
        float c = dot(ro, ro) - radius * radius;
        float D = b * b - 4 * a * c; //discriminant
        //number of roots/intersections: 0 when D < 0; 1 when D = 0; 2 when D > 0
        if(D > 0){
            float sD = std::sqrt(D);
            float t_near = (-b - sD) / (2.0 * a);
            float t_far = (-b + sD) / (2.0 * a);
            bool outside = 0 <= t_near;
            float t = outside? t_near : t_far;
            if(t >= t_min && t <= t_max) {
                result.t = t;
                auto hit_point = ro + t*rd;
                result.normal = normalize(hit_point);
                if(!outside) result.normal = -result.normal;
                auto uv_map = vec(cartesian_to_hyperspherical(vec_ref({hit_point[1], hit_point[0], hit_point[2]})));
                result.uv = vec_ref({uv_map[2]/(std::numbers::pi_v<float>*2), uv_map[1]/std::numbers::pi_v<float>});
                result.hit = true;
            }
        }

        return result;
    }
};

struct Plane : Shape {
    float a = 0, b = 1, c = 0, d = 0;

    Plane() = default;
    Plane(float a, float b, float c, float d) : a(a), b(b), c(c), d(d) {}

    IntersectionResult Trace(const vec3& ro, const vec3& rd, float t_min, float t_max) {
        IntersectionResult result;
        float vro = dot(vec_ref({a, b, c}), ro);
        float vrd = dot(vec_ref({a, b, c}), rd);
        float t = (-d - vro)/vrd;
        if(t >= t_min && t <= t_max){
            result.t = t;
            result.normal = normalize(vec_ref({a, b, c}));
            vec3 plane_space_hit_point = transpose(tangent_space_from_normal(result.normal))*(ro + t*rd);
            result.uv[0] = plane_space_hit_point[0];
            result.uv[1] = plane_space_hit_point[2];
            result.hit = true;
        }

        return result;
    }
};

void ray_transform(const mat4& inv_world, vec3& ray_origin, vec3& ray_direction) {
    //transform ray to object space
    ray_origin = inv_world * join(ray_origin, 1);
    ray_direction = inv_world * join(ray_direction, 0);
}

struct Transform {
	vec3 translation {0, 0, 0};
	vec3 scaling {1, 1, 1};
	quat_t<float> rotation = quat(vec_ref<float>({1, 0, 0, 0}));
	
	mat4 Compose() {
        return
            Maths::translation(translation) *
            Maths::as_matrix<4,4>(rotation) *
            Maths::scaling(join(scaling, 1));
    }
};

struct Material {
    Material() = default;
    virtual ~Material() = default;

    //overly-simplified model that can only redirect paths along
    //one deterministic direction in the spirit of whitted-style path tracing
    //returns boolean value which when set to false terminates the path
    virtual bool Evaluate(const IntersectionResult& ires, vec3& ray_origin, vec3& ray_direction, vec3& throughput, int& refraction_count) = 0;
};

struct DiffuseMaterial : Material {
    vec3 tint;
    DiffuseMaterial(const vec3& tint) : tint(tint) {};

    bool Evaluate(const IntersectionResult& ires, vec3& ray_origin, vec3& ray_direction, vec3& throughput, int& refraction_count) {
        throughput *= tint;
        //stop building the path and do the final shading
        return false;
    }
};

struct MirrorMaterial : Material {
    vec3 tint;
    MirrorMaterial(const vec3& tint) : tint(tint) {};

    bool Evaluate(const IntersectionResult& ires, vec3& ray_origin, vec3& ray_direction, vec3& throughput, int& refraction_count) {
        throughput *= tint;
        //reflect the ray along normal
        ray_origin = ray_origin + ray_direction * ires.t + ires.normal * 0.0001f;
        ray_direction = reflect<Conventions::RayDirection::Incident>(ray_direction, ires.normal);
        return true;
    }
};

struct GlassMaterial : Material {
    vec3 tint;

    GlassMaterial(const vec3& tint) : tint(tint) {};

    bool Evaluate(const IntersectionResult& ires, vec3& ray_origin, vec3& ray_direction, vec3& throughput, int& refraction_count) {
        //refract the ray along normal
        ray_origin = ray_origin + ray_direction * ires.t - ires.normal * 0.0001f;
        constexpr bool TotalInternalReflection = true;
        constexpr float IndexOfRefractionAir = 1.0;
        constexpr float IndexOfRefractionGlass = 1.5;
        //odd-even test to determine whether we are entering or exiting the solid
        bool outside = refraction_count%2;
        if(!outside) throughput *= tint;
        ray_direction = refract<Conventions::RayDirection::Incident, TotalInternalReflection>(
            ray_direction,
            ires.normal,
            outside?IndexOfRefractionGlass:IndexOfRefractionAir,
            outside?IndexOfRefractionAir:IndexOfRefractionGlass
        );
        ++refraction_count;
        return true;
    }
};

struct Object {
    std::unique_ptr<Shape> shape;
    std::unique_ptr<Material> material;
    Transform transform;
    mat4 inv_world;

    Object(std::unique_ptr<Shape>&& shape, std::unique_ptr<Material>&& material)
     : shape(std::move(shape)), material(std::move(material))
    {}
    ~Object() = default;

    void Update() {
        inv_world = inverse(transform.Compose());
    }

    IntersectionResult Trace(vec3 ray_origin, vec3 ray_direction, float t_min, float t_max) {
        ray_transform(inv_world, ray_origin, ray_direction);
        IntersectionResult ret = shape->Trace(ray_origin, ray_direction, t_min, t_max);
        if(ret.hit) ret.normal = normalize(vec3(transpose(inv_world) * join(ret.normal, 0)));
        return ret;
    }
};

struct World {
    std::vector<std::unique_ptr<Object>> objects;

    World() = default;
    ~World() = default;

    void AddObject(std::unique_ptr<Object>&& object) {
        objects.push_back(std::move(object));
    }

    void Update() {
        for(auto&& obj : objects) {
            obj->Update();
        }
    }

    IntersectionResult TraceClosest(vec3 ray_origin, vec3 ray_direction, float t_min, float t_max, Material*& hit_mat) {
        IntersectionResult ires_closest;
        //in a real application a top-level acceleration data structure is employed instead of linear search
        for(auto&& obj : objects) {
            IntersectionResult ires = obj->Trace(ray_origin, ray_direction, 0.0f, std::numeric_limits<float>::max());
            if(!ires_closest.hit || ires.t < ires_closest.t) {
                ires_closest = ires;
                hit_mat = obj->material.get();
            }
        }
        return ires_closest;
    }

    IntersectionResult TraceAny(vec3 ray_origin, vec3 ray_direction, float t_min, float t_max) {
        IntersectionResult ires;
        for(auto&& obj : objects) {
            ires = obj->Trace(ray_origin, ray_direction, 0.0f, std::numeric_limits<float>::max());
            if(ires.hit) break;
        }
        return ires;
    }
};

class Timer {
public:
    Timer() : beg_(clock_::now()) {}
    void reset() { beg_ = clock_::now(); }
    double elapsed() const { 
        return std::chrono::duration_cast<second_>
            (clock_::now() - beg_).count(); }

private:
    typedef std::chrono::high_resolution_clock clock_;
    typedef std::chrono::duration<double, std::ratio<1> > second_;
    std::chrono::time_point<clock_> beg_;
};

int main(int argc, char ** argv) {
	bool quit = false;
    mat_dynamic_t<Uint32> image_buffer(600, 800);

    SDL_Init(SDL_INIT_VIDEO);

    SDL_Window * window = SDL_CreateWindow(
        "Ray Tracing Sample",
        SDL_WINDOWPOS_UNDEFINED,
        SDL_WINDOWPOS_UNDEFINED,
        image_buffer.column_count().get(),
        image_buffer.row_count().get(),
        0
    );

    SDL_Renderer * renderer = SDL_CreateRenderer(window, -1, 0);
	SDL_Texture * texture = SDL_CreateTexture(
        renderer,
        SDL_PIXELFORMAT_ARGB8888,
        SDL_TEXTUREACCESS_STATIC,
        image_buffer.column_count().get(),
        image_buffer.row_count().get()
    );
	SDL_PixelFormat* pxfmt = SDL_AllocFormat(SDL_PIXELFORMAT_ARGB8888);

    //translation along x, y, z axes
    vec3 camera_position {0, 0, 2};
    //rotation around x, y, z axes
    vec3 camera_rotation {0, 0, 0};

    auto proj = projection(std::numbers::pi_v<float>/2.0, static_cast<float>(image_buffer.column_count().get())/image_buffer.row_count().get(), 0.01, 100.0);

    bvh::v2::ThreadPool pool;

    World world;
    world.AddObject(
        std::make_unique<Object>(
            std::make_unique<AABB>(
                vec3{-1, -1, -1},
                vec3{1, 1, 1}
            ),
            std::make_unique<DiffuseMaterial>(
                vec3{1.0f, 0.7f, 0.4f}
            )
        )
    );
    world.AddObject(
        std::make_unique<Object>(
            std::make_unique<Sphere>(1.0f),
            std::make_unique<MirrorMaterial>(
                vec3{1, 1, 1}
            )
        )
    );
    world.AddObject(
        std::make_unique<Object>(
            std::make_unique<Plane>(0, 1, 0, 0),
            std::make_unique<DiffuseMaterial>(
                vec3{0.9f, 1.0f, 0.9f}
            )
        )
    );
    world.objects.back()->transform.translation[1] = -2;
    world.objects.back()->transform.scaling[0] = 10.0f;
    world.objects.back()->transform.scaling[2] = 10.0f;
    world.AddObject(
        std::make_unique<Object>(
            std::make_unique<Sphere>(1.0f),
            std::make_unique<GlassMaterial>(
                vec3{0.7f, 0.9f, 1.0f}
            )
        )
    );
    world.objects.back()->transform.translation[2] = -7;

    Timer timer;
    float time_elapsed = timer.elapsed();

    while (!quit) {
        //update time variables
        float time_elapsed_previous = time_elapsed;
        time_elapsed = timer.elapsed();
        float time_delta = time_elapsed - time_elapsed_previous;

        //handle SDL2 events
		SDL_Event event;
		while(SDL_PollEvent(&event)) {
			switch (event.type) {
				case SDL_QUIT: quit = true; break;
			}
		}

        //handle mouse controls and compute orientation
        int mouse_delta_x = 0, mouse_delta_y = 0;
        SDL_GetRelativeMouseState(&mouse_delta_x, &mouse_delta_y);
        if(SDL_GetMouseState(nullptr, nullptr) & SDL_BUTTON_LMASK) {
            camera_rotation += vec3{static_cast<float>(-mouse_delta_y), static_cast<float>(-mouse_delta_x), 0} * 0.005;
        }
        auto rotation = quat(quat_axis_angle(vec3{0, 1, 0}, camera_rotation[1]) * quat_axis_angle(vec3{1, 0, 0}, camera_rotation[0]));

        //handle keyboard controls and update position
        const Uint8* keystate = SDL_GetKeyboardState(nullptr);
        float speed = 10.0f;
        if (keystate[SDL_SCANCODE_LCTRL]) {
			speed *= 10.0f;
		}
        if (keystate[SDL_SCANCODE_LSHIFT]) {
			speed *= 0.1f;
		}
        speed *= time_delta;
        if (keystate[SDL_SCANCODE_A]) {
			camera_position += rotation * vec3{-speed, 0, 0};
		}
		if (keystate[SDL_SCANCODE_D]) {
			camera_position += rotation * vec3{speed, 0, 0};
		}
		if (keystate[SDL_SCANCODE_Q]) {
			camera_position += rotation * vec3{0, -speed, 0};
		}
		if (keystate[SDL_SCANCODE_E]) {
			camera_position += rotation * vec3{0, speed, 0};
		}
		if (keystate[SDL_SCANCODE_W]) {
			camera_position += rotation * vec3{0, 0, -speed};
		}
		if (keystate[SDL_SCANCODE_S]) {
			camera_position += rotation * vec3{0, 0, speed};
		}

        //view matrix is inverted because from the camera's point of view everything is transforming in the opposite way
        mat4 view = inverse(translation(camera_position) * as_matrix<4,4>(rotation));
        //and this matrix is inverted to do unprojection from normalized device coordinates back to world space
        mat4 inv_viewproj = inverse(proj*view);

        //update world objects
        world.objects[0]->transform.rotation *= quat_axis_angle(vec3{0, 1, 0}, std::numbers::pi_v<float> * 2.0f * time_delta);
        world.objects[0]->transform.translation[0] = std::cos(std::numbers::pi_v<float> * 2.0f * 0.1f * time_elapsed) * 4.0f;
        world.objects[0]->transform.translation[1] = std::cos(std::numbers::pi_v<float> * 2.0f * 0.5f * time_elapsed);
        world.objects[0]->transform.translation[2] = std::sin(std::numbers::pi_v<float> * 2.0f * 0.1f * time_elapsed) * 4.0f;
        world.objects[1]->transform.scaling[0] = std::sin(std::numbers::pi_v<float> * 2.0f * 0.1f * time_elapsed) * 4.0f;
        world.Update();

        auto image_size = ivec2{
            static_cast<int>(image_buffer.column_count().get()),
            static_cast<int>(image_buffer.row_count().get())
        };
        //set fixed tile size
        const int threads_per_dim = std::sqrt(std::thread::hardware_concurrency());
        const auto tile_size = image_size/threads_per_dim;
        //calculate horizontal and vertical tile count
		ivec2 tile_count {
            static_cast<int>(image_size[0] / tile_size[0] + (image_size[0] % tile_size[0] > 0)),
            static_cast<int>(image_size[1] / tile_size[1] + (image_size[1] % tile_size[1] > 0))
        };
        //iterate each tile
		for (int i = 0; i < tile_count[1]; ++i) {
			for (int j = 0; j < tile_count[0]; ++j) {
                //schedule a thread per tile
				pool.push([&, i, j](size_t thread_id) {
                    //use block matrix partitioner to create a matrix for access to this tile
                    auto tile_origin = ivec2{j*tile_size[0], i*tile_size[1]};
                    auto tile = rectangular_partition(image_buffer, tile_origin[1], tile_size[1], tile_origin[0], tile_size[0]);
                    //perform operation on the tile matrix
                    tile = unary_operation<true>(
                        tile,
                        [&](auto val, auto tile_y, auto tile_x) {
                            auto tile_pos = vec<int>(vec_ref({tile_x, tile_y}));
                            auto image_pos = tile_origin + tile_pos;
                            auto uv = vec2(vec<float>(image_pos)/image_size); uv[1] = 1-uv[1];
                            auto ray_origin = vec(homogeneous_division(
                                inv_viewproj * join(uv*2-1, vec2{0, 1})
                            ));
                            auto ray_direction = vec(normalize(homogeneous_division(
                                inv_viewproj * join(uv*2-1, vec2{1, 1})
                            ) - ray_origin));

                            vec3 color {0, 0, 0};
                            vec3 throughput {1, 1, 1};
                            IntersectionResult ires_closest;
                            int refraction_count = 0;
                            float distance_traveled_in_air = 0.0f;

                            //trace a path with a fixed maximum amount of vertices between each line-segment (ray)
                            for(int path_vertex = 0; path_vertex < 5; ++path_vertex) {
                                Material* ptr_hit_mat = nullptr;
                                ires_closest = world.TraceClosest(ray_origin, ray_direction, 0.0f, std::numeric_limits<float>::max(), ptr_hit_mat);
                                if(ires_closest.hit) {
                                    auto hit_point = ray_origin + ray_direction * ires_closest.t;
                                    if(!(refraction_count%2)) distance_traveled_in_air += length(hit_point - ray_origin);
                                    if(ptr_hit_mat->Evaluate(ires_closest, ray_origin, ray_direction, throughput, refraction_count)) continue;
                                    color = as_vector<3>(checker_pattern(ires_closest.uv, 4.0f));
                                    color *= throughput;
                                    auto light_direction = vec(normalize(vec_ref<float>({1, 2, 3})));
                                    auto hit_point_shifted = hit_point + ires_closest.normal * 0.001f;
                                    color *= as_vector<3, float>(std::clamp(dot(ires_closest.normal, light_direction), 0.0f, 1.0f));
                                    //trace shadow
                                    color *= world.TraceAny(hit_point_shifted, light_direction, 0.0f, std::numeric_limits<float>::max()).hit? 0.1f : 1.0f;
                                    //fog absorption
                                    color *= beer_lambert_law(0.05f, distance_traveled_in_air);
                                    //color = (ires_closest.normal+1)/2;
                                    //color = join(ires_closest.uv, 0);
                                    break;
                                }
                                //sky
                                color = vec_ref({0.4f, 0.7f, 1.0f}) * std::clamp(dot(ray_direction, vec_ref({0, 1, 0})), 0.0f, 1.0f);
                                break;
                            }

                            return SDL_MapRGBA(
                                pxfmt,
                                color[0]*255,
                                color[1]*255,
                                color[2]*255,
                                255
                            );
                        }
                    );
				});
			}
		}
		pool.wait();
        
        SDL_UpdateTexture(texture, NULL, image_buffer.data.data(), image_buffer.column_count().get() * sizeof(Uint32));

        SDL_RenderClear(renderer);
		SDL_RenderCopy(renderer, texture, NULL, NULL);
		SDL_RenderPresent(renderer);
    }

    SDL_FreeFormat(pxfmt);
    
	SDL_DestroyTexture(texture);
	SDL_DestroyRenderer(renderer);
    
    SDL_DestroyWindow(window);
	SDL_Quit();
	
    return EXIT_SUCCESS;
}
