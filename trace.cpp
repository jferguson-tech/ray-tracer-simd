// CPU Pathtracer with Demo Recording & Benchmarking
// With Physically-Based Caustics


#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <thread>
#include <atomic>
#include <chrono>
#include <algorithm>
#include <cstdio>
#include <sstream>
#include <iomanip>
#include <cstring>
#include <queue>
#include <set>
#include <immintrin.h>
#include <SDL2/SDL.h>
#include <fstream>
#include <memory>

// Platform-specific includes for system info
#ifdef _WIN32
#include <windows.h>
#include <intrin.h>
#else
#include <unistd.h>
#include <sys/utsname.h>
#include <cpuid.h>
#endif

// Simple JSON writer
class JSONWriter {
    std::ostringstream ss;
    std::vector<bool> firstInScope;
    int indent = 0;
    
    void writeIndent() {
        for (int i = 0; i < indent; i++) ss << "  ";
    }
    
public:
    void startObject(const std::string& key = "") {
        if (!key.empty()) {
            if (!firstInScope.back()) ss << ",";
            firstInScope.back() = false;
            ss << "\n";
            writeIndent();
            ss << "\"" << key << "\": {";
        } else {
            if (!firstInScope.empty() && !firstInScope.back()) ss << ",";
            if (!firstInScope.empty()) firstInScope.back() = false;
            ss << "\n";
            writeIndent();
            ss << "{";
        }
        indent++;
        firstInScope.push_back(true);
    }
    
    void endObject() {
        indent--;
        firstInScope.pop_back();
        ss << "\n";
        writeIndent();
        ss << "}";
    }
    
    void startArray(const std::string& key) {
        if (!firstInScope.back()) ss << ",";
        firstInScope.back() = false;
        ss << "\n";
        writeIndent();
        ss << "\"" << key << "\": [";
        indent++;
        firstInScope.push_back(true);
    }
    
    void endArray() {
        indent--;
        firstInScope.pop_back();
        ss << "\n";
        writeIndent();
        ss << "]";
    }
    
    void addString(const std::string& key, const std::string& value) {
        if (!firstInScope.back()) ss << ",";
        firstInScope.back() = false;
        ss << "\n";
        writeIndent();
        ss << "\"" << key << "\": \"" << value << "\"";
    }
    
    void addNumber(const std::string& key, double value) {
        if (!firstInScope.back()) ss << ",";
        firstInScope.back() = false;
        ss << "\n";
        writeIndent();
        ss << "\"" << key << "\": " << value;
    }
    
    void addBool(const std::string& key, bool value) {
        if (!firstInScope.back()) ss << ",";
        firstInScope.back() = false;
        ss << "\n";
        writeIndent();
        ss << "\"" << key << "\": " << (value ? "true" : "false");
    }
    
    std::string toString() { return ss.str(); }
};

// System Information
struct SystemInfo {
    std::string cpuModel;
    int cpuCores;
    int cpuThreads;
    std::string osName;
    std::string compilerInfo;
    
    static SystemInfo get() {
        SystemInfo info;
        
#ifdef _WIN32
        // Windows system info
        SYSTEM_INFO sysInfo;
        GetSystemInfo(&sysInfo);
        info.cpuCores = sysInfo.dwNumberOfProcessors;
        info.cpuThreads = std::thread::hardware_concurrency();
        
        // Get CPU name
        int cpuInfo[4] = {0};
        char cpuBrand[0x40] = {0};
        __cpuid(cpuInfo, 0x80000002);
        memcpy(cpuBrand, cpuInfo, sizeof(cpuInfo));
        __cpuid(cpuInfo, 0x80000003);
        memcpy(cpuBrand + 16, cpuInfo, sizeof(cpuInfo));
        __cpuid(cpuInfo, 0x80000004);
        memcpy(cpuBrand + 32, cpuInfo, sizeof(cpuInfo));
        info.cpuModel = std::string(cpuBrand);
        
        info.osName = "Windows";
#else
        // Linux/Unix system info
        info.cpuCores = sysconf(_SC_NPROCESSORS_ONLN);
        info.cpuThreads = std::thread::hardware_concurrency();
        
        // Get CPU name from /proc/cpuinfo
        std::ifstream cpuinfo("/proc/cpuinfo");
        std::string line;
        while (std::getline(cpuinfo, line)) {
            if (line.find("model name") != std::string::npos) {
                size_t pos = line.find(':');
                if (pos != std::string::npos) {
                    info.cpuModel = line.substr(pos + 2);
                    break;
                }
            }
        }
        
        struct utsname unameData;
        if (uname(&unameData) == 0) {
            info.osName = std::string(unameData.sysname) + " " + unameData.release;
        }
#endif
        
        // Compiler info
#ifdef __GNUC__
        info.compilerInfo = "GCC " + std::to_string(__GNUC__) + "." + std::to_string(__GNUC_MINOR__);
#elif defined(_MSC_VER)
        info.compilerInfo = "MSVC " + std::to_string(_MSC_VER);
#else
        info.compilerInfo = "Unknown";
#endif
        
        return info;
    }
    
    void toJSON(JSONWriter& json) {
        json.addString("cpu_model", cpuModel);
        json.addNumber("cpu_cores", cpuCores);
        json.addNumber("cpu_threads", cpuThreads);
        json.addString("os", osName);
        json.addString("compiler", compilerInfo);
    }
};

// Camera keyframe for recording
struct CameraKeyframe {
    float time;
    float x, y, z;
    float yaw, pitch;
    
    CameraKeyframe(float t, float x, float y, float z, float yaw, float pitch)
        : time(t), x(x), y(y), z(z), yaw(yaw), pitch(pitch) {}
};

// Demo path recorder/player
class DemoPath {
public:
    std::vector<CameraKeyframe> keyframes;
    float totalDuration = 0;
    
    void addKeyframe(float time, float x, float y, float z, float yaw, float pitch) {
        keyframes.emplace_back(time, x, y, z, yaw, pitch);
        totalDuration = std::max(totalDuration, time);
    }
    
    void clear() {
        keyframes.clear();
        totalDuration = 0;
    }
    
    bool getInterpolatedCamera(float time, float& x, float& y, float& z, float& yaw, float& pitch) const {
        if (keyframes.empty()) return false;
        
        // Loop the demo
        time = std::fmod(time, totalDuration);
        if (time < 0) time += totalDuration;
        
        // Find the two keyframes to interpolate between
        size_t i = 0;
        for (; i < keyframes.size() - 1; i++) {
            if (keyframes[i + 1].time > time) break;
        }
        
        if (i >= keyframes.size() - 1) {
            // Use last keyframe
            const auto& kf = keyframes.back();
            x = kf.x; y = kf.y; z = kf.z;
            yaw = kf.yaw; pitch = kf.pitch;
            return true;
        }
        
        // Interpolate between keyframes[i] and keyframes[i+1]
        const auto& kf1 = keyframes[i];
        const auto& kf2 = keyframes[i + 1];
        
        float t = (time - kf1.time) / (kf2.time - kf1.time);
        
        // Smooth interpolation using cubic ease
        t = t * t * (3.0f - 2.0f * t);
        
        x = kf1.x + (kf2.x - kf1.x) * t;
        y = kf1.y + (kf2.y - kf1.y) * t;
        z = kf1.z + (kf2.z - kf1.z) * t;
        
        // Interpolate angles correctly
        float yawDiff = kf2.yaw - kf1.yaw;
        if (yawDiff > M_PI) yawDiff -= 2 * M_PI;
        if (yawDiff < -M_PI) yawDiff += 2 * M_PI;
        yaw = kf1.yaw + yawDiff * t;
        
        pitch = kf1.pitch + (kf2.pitch - kf1.pitch) * t;
        
        return true;
    }
    
    void saveToFile(const std::string& filename) const {
        JSONWriter json;
        json.startObject();
        json.addNumber("total_duration", totalDuration);
        json.addNumber("keyframe_count", keyframes.size());
        json.startArray("keyframes");
        
        for (const auto& kf : keyframes) {
            json.startObject();
            json.addNumber("time", kf.time);
            json.addNumber("x", kf.x);
            json.addNumber("y", kf.y);
            json.addNumber("z", kf.z);
            json.addNumber("yaw", kf.yaw);
            json.addNumber("pitch", kf.pitch);
            json.endObject();
        }
        
        json.endArray();
        json.endObject();
        
        std::ofstream file(filename);
        file << json.toString();
        file.close();
        
        std::cout << "Saved demo path to " << filename << " (" << keyframes.size() << " keyframes)\n";
    }
    
    bool loadFromFile(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open " << filename << "\n";
            return false;
        }
        
        // Simple JSON parser
        std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        file.close();
        
        clear();
        
        // Parse total_duration
        size_t pos = content.find("\"total_duration\":");
        if (pos != std::string::npos) {
            pos += 17;
            totalDuration = std::stof(content.substr(pos));
        }
        
        // Parse keyframes
        pos = content.find("\"keyframes\":");
        if (pos != std::string::npos) {
            pos = content.find("[", pos);
            size_t endPos = content.find("]", pos);
            
            size_t kfPos = pos;
            while ((kfPos = content.find("{", kfPos + 1)) < endPos) {
                float time, x, y, z, yaw, pitch;
                
                size_t timePos = content.find("\"time\":", kfPos);
                time = std::stof(content.substr(timePos + 7));
                
                size_t xPos = content.find("\"x\":", kfPos);
                x = std::stof(content.substr(xPos + 4));
                
                size_t yPos = content.find("\"y\":", kfPos);
                y = std::stof(content.substr(yPos + 4));
                
                size_t zPos = content.find("\"z\":", kfPos);
                z = std::stof(content.substr(zPos + 4));
                
                size_t yawPos = content.find("\"yaw\":", kfPos);
                yaw = std::stof(content.substr(yawPos + 6));
                
                size_t pitchPos = content.find("\"pitch\":", kfPos);
                pitch = std::stof(content.substr(pitchPos + 8));
                
                keyframes.emplace_back(time, x, y, z, yaw, pitch);
                
                kfPos = content.find("}", kfPos);
            }
        }
        
        std::cout << "Loaded demo path from " << filename << " (" << keyframes.size() << " keyframes)\n";
        return true;
    }
};

// Benchmark data
struct BenchmarkFrame {
    float time;
    int fps;
    int samples;
    float renderTime;
};

class BenchmarkRecorder {
public:
    std::vector<BenchmarkFrame> frames;
    SystemInfo systemInfo;
    int renderWidth, renderHeight;
    float totalTime = 0;
    
    void recordFrame(float time, int fps, int samples, float renderTime) {
        frames.push_back({time, fps, samples, renderTime});
    }
    
    void saveResults(const std::string& filename) {
        JSONWriter json;
        json.startObject();
        
        // System info as nested object
        json.startObject("system_info");
        systemInfo.toJSON(json);
        json.endObject();
        
        // Benchmark settings
        json.addNumber("render_width", renderWidth);
        json.addNumber("render_height", renderHeight);
        json.addNumber("total_time", totalTime);
        json.addNumber("total_frames", frames.size());
        
        // Calculate statistics
        if (!frames.empty()) {
            float avgFPS = 0, minFPS = frames[0].fps, maxFPS = frames[0].fps;
            for (const auto& f : frames) {
                avgFPS += f.fps;
                minFPS = std::min(minFPS, (float)f.fps);
                maxFPS = std::max(maxFPS, (float)f.fps);
            }
            avgFPS /= frames.size();
            
            json.addNumber("avg_fps", avgFPS);
            json.addNumber("min_fps", minFPS);
            json.addNumber("max_fps", maxFPS);
        }
        
        // Frame data
        json.startArray("frames");
        for (const auto& f : frames) {
            json.startObject();
            json.addNumber("time", f.time);
            json.addNumber("fps", f.fps);
            json.addNumber("samples", f.samples);
            json.addNumber("render_time_ms", f.renderTime);
            json.endObject();
        }
        json.endArray();
        
        json.endObject();
        
        std::ofstream file(filename);
        file << json.toString();
        file.close();
        
        std::cout << "Saved benchmark results to " << filename << "\n";
    }
};

// Configuration with new modes - Updated to 16:9 resolutions
struct Settings {
    int renderWidth = 640;
    int renderHeight = 360;
    int windowWidth = 1280;
    int windowHeight = 720;
    int worldSeed = 42;
    float timeOfDay = 0.85f;
    float waterAnimation = 0.0f;
    bool showUI = true;
    bool enableCaustics = true;
    bool enableVolumetrics = true;
    int causticQuality = 3;  // 1=low (8 samples), 2=medium (16 samples), 3=high (32 samples)
    
    // New settings for recording/playback
    enum Mode {
        MODE_INTERACTIVE,
        MODE_RECORDING,
        MODE_PLAYBACK,
        MODE_BENCHMARK,
        MODE_OFFLINE_RENDER
    } mode = MODE_INTERACTIVE;
    
    int offlineTargetSamples = 1000;  // For offline rendering
    std::string outputDir = "output";
    
    void adjustRenderResolution(int preset) {
        switch(preset) {
            case 1: renderWidth = 256; renderHeight = 144; break;   // 16:9 (144p)
            case 2: renderWidth = 426; renderHeight = 240; break;   // 16:9 (240p)
            case 3: renderWidth = 640; renderHeight = 360; break;   // 16:9 (360p)
            case 4: renderWidth = 854; renderHeight = 480; break;   // 16:9 (480p)
            case 5: renderWidth = 1280; renderHeight = 720; break;  // 16:9 (720p HD)
            case 6: renderWidth = 1920; renderHeight = 1080; break; // 16:9 (1080p Full HD)
        }
    }
    
    void adjustWindowSize(bool increase) {
        float scale = windowWidth / float(renderWidth);
        if (increase && scale < 6.0f) {
            scale += 0.5f;
        } else if (!increase && scale > 1.0f) {
            scale -= 0.5f;
        }
        windowWidth = renderWidth * scale;
        windowHeight = renderHeight * scale;
    }
};

Settings g_settings;

// Constants
constexpr int WORLD_SIZE = 128;
constexpr int WORLD_HEIGHT = 48;
constexpr int MAX_BOUNCES = 5;
constexpr int SAMPLES_PER_PIXEL = 2;
constexpr float FOV = 90.0f;
constexpr float MAX_RAY_DISTANCE = 500.0f;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Math utilities
struct Vec3 {
    float x, y, z;
    Vec3() : x(0), y(0), z(0) {}
    Vec3(float x, float y, float z) : x(x), y(y), z(z) {}
    
    Vec3 operator+(const Vec3& v) const { return Vec3(x + v.x, y + v.y, z + v.z); }
    Vec3 operator-(const Vec3& v) const { return Vec3(x - v.x, y - v.y, z - v.z); }
    Vec3 operator*(float t) const { return Vec3(x * t, y * t, z * t); }
    Vec3 operator*(const Vec3& v) const { return Vec3(x * v.x, y * v.y, z * v.z); }
    Vec3 operator/(float t) const { return Vec3(x / t, y / t, z / t); }
    Vec3 operator-() const { return Vec3(-x, -y, -z); }
    
    float dot(const Vec3& v) const { return x * v.x + y * v.y + z * v.z; }
    Vec3 cross(const Vec3& v) const {
        return Vec3(y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x);
    }
    
    float length() const { return std::sqrt(x * x + y * y + z * z); }
    Vec3 normalize() const { 
        float l = length(); 
        return l > 0 ? *this / l : Vec3(); 
    }
    
    Vec3& operator+=(const Vec3& v) { 
        x += v.x; y += v.y; z += v.z; 
        return *this; 
    }
    
    bool near_zero() const {
        const float s = 1e-8f;
        return (std::abs(x) < s) && (std::abs(y) < s) && (std::abs(z) < s);
    }
};

struct Vec3i {
    int x, y, z;
    Vec3i(int x, int y, int z) : x(x), y(y), z(z) {}
    Vec3i operator+(const Vec3i& v) const { return Vec3i(x + v.x, y + v.y, z + v.z); }
};

// Ray structure
struct Ray {
    Vec3 origin;
    Vec3 direction;
    
    Ray(const Vec3& o, const Vec3& d) : origin(o), direction(d.normalize()) {}
    Vec3 at(float t) const { return origin + direction * t; }
};

// Block types
enum BlockType : uint8_t {
    AIR = 0,
    STONE,
    GRASS,
    DIRT,
    WOOD,
    LEAVES,
    LIGHT,
    WATER,
    SAND
};

// Static material properties
struct MaterialProps {
    Vec3 albedo;
    Vec3 emission;
    float roughness;
    float ior;
    float transparency;
    bool isVolume;
};

// Static material table
static const MaterialProps g_materials[] = {
    {{0, 0, 0}, {0, 0, 0}, 1.0f, 1.0f, 0.0f, false},           // AIR
    {{0.5f, 0.5f, 0.5f}, {0, 0, 0}, 0.8f, 1.0f, 0.0f, false},  // STONE
    {{0.2f, 0.6f, 0.2f}, {0, 0, 0}, 0.9f, 1.0f, 0.0f, false},  // GRASS
    {{0.4f, 0.3f, 0.2f}, {0, 0, 0}, 0.9f, 1.0f, 0.0f, false},  // DIRT
    {{0.6f, 0.4f, 0.2f}, {0, 0, 0}, 0.7f, 1.0f, 0.0f, false},  // WOOD
    {{0.3f, 0.7f, 0.3f}, {0, 0, 0}, 0.8f, 1.0f, 0.0f, false},  // LEAVES
    {{1, 1, 1}, {10, 10, 8}, 0.1f, 1.0f, 0.0f, false},         // LIGHT
    {{0.1f, 0.35f, 0.45f}, {0, 0, 0}, 0.02f, 1.333f, 0.65f, true}, // WATER
    {{0.76f, 0.7f, 0.5f}, {0, 0, 0}, 0.9f, 1.0f, 0.0f, false}  // SAND
};

// Sun light system
struct SunLight {
    Vec3 direction;
    Vec3 color;
    float intensity;
    
    void updateFromTimeOfDay(float timeOfDay) {
        float sunAngle = timeOfDay * M_PI;
        
        direction = Vec3(
            -std::cos(sunAngle),
            -std::sin(sunAngle) * 0.8f - 0.2f,
            0.0f
        ).normalize();
        
        if (timeOfDay < 0.25f) {
            float t = timeOfDay * 4.0f;
            color = Vec3(1.0f, 0.6f, 0.3f) * t + Vec3(0.2f, 0.2f, 0.3f) * (1 - t);
            intensity = 0.2f + 0.6f * t;
        } else if (timeOfDay < 0.75f) {
            color = Vec3(1.0f, 0.95f, 0.8f);
            intensity = 0.8f + 0.2f * std::sin((timeOfDay - 0.25f) * 2 * M_PI);
        } else {
            float t = (timeOfDay - 0.75f) * 4.0f;
            color = Vec3(1.0f, 0.95f, 0.8f) * (1 - t) + Vec3(1.0f, 0.5f, 0.3f) * t;
            intensity = 0.8f * (1 - t) + 0.2f * t;
        }
    }
    
    Vec3 getLightContribution() const {
        return color * intensity;
    }
};

// Random utilities
thread_local std::mt19937 rng(std::random_device{}());
thread_local std::uniform_real_distribution<float> dist(0.0f, 1.0f);

inline float random01() { return dist(rng); }

inline Vec3 randomInHemisphere(const Vec3& normal) {
    Vec3 dir;
    do {
        dir = Vec3(random01() * 2 - 1, random01() * 2 - 1, random01() * 2 - 1);
    } while (dir.dot(dir) > 1);
    
    dir = dir.normalize();
    if (dir.dot(normal) < 0) dir = dir * -1;
    return dir;
}

// Simple hash function for procedural noise
inline float hash(float x, float y, float z) {
    float n = std::sin(x * 12.9898f + y * 78.233f + z * 37.719f) * 43758.5453f;
    return n - std::floor(n);
}

// Trilinear interpolation
inline float lerp(float a, float b, float t) {
    return a + t * (b - a);
}

// Smooth interpolation curve
inline float smoothstep(float t) {
    return t * t * (3.0f - 2.0f * t);
}

// Simple 3D value noise
inline float noise3D(float x, float y, float z) {
    float ix = std::floor(x);
    float iy = std::floor(y);
    float iz = std::floor(z);
    
    float fx = x - ix;
    float fy = y - iy;
    float fz = z - iz;
    
    // Get noise values at cube corners
    float n000 = hash(ix, iy, iz);
    float n100 = hash(ix + 1, iy, iz);
    float n010 = hash(ix, iy + 1, iz);
    float n110 = hash(ix + 1, iy + 1, iz);
    float n001 = hash(ix, iy, iz + 1);
    float n101 = hash(ix + 1, iy, iz + 1);
    float n011 = hash(ix, iy + 1, iz + 1);
    float n111 = hash(ix + 1, iy + 1, iz + 1);
    
    // Smooth the fractional parts
    float sx = smoothstep(fx);
    float sy = smoothstep(fy);
    float sz = smoothstep(fz);
    
    // Trilinear interpolation
    float nx00 = lerp(n000, n100, sx);
    float nx10 = lerp(n010, n110, sx);
    float nx01 = lerp(n001, n101, sx);
    float nx11 = lerp(n011, n111, sx);
    
    float nxy0 = lerp(nx00, nx10, sy);
    float nxy1 = lerp(nx01, nx11, sy);
    
    return lerp(nxy0, nxy1, sz);
}

// Fractal Brownian Motion (fBm) - combines multiple octaves of noise
inline float fbm(float x, float y, float z, int octaves = 4) {
    float value = 0.0f;
    float amplitude = 0.5f;
    float frequency = 1.0f;
    
    for (int i = 0; i < octaves; i++) {
        value += amplitude * noise3D(x * frequency, y * frequency, z * frequency);
        amplitude *= 0.5f;
        frequency *= 2.0f;
    }
    
    return value;
}

// Procedural dirt texture
inline Vec3 getDirtTexture(const Vec3& pos, const Vec3& normal) {
    // Base dirt color
    Vec3 baseColor(0.4f, 0.3f, 0.2f);
    
    // Large scale color variation (soil patches)
    float largeNoise = fbm(pos.x * 0.2f, pos.y * 0.2f, pos.z * 0.2f, 3);
    Vec3 darkSoil(0.25f, 0.18f, 0.12f);
    Vec3 lightSoil(0.5f, 0.38f, 0.28f);
    Vec3 soilColor = darkSoil * (1.0f - largeNoise) + lightSoil * largeNoise;
    
    // Medium scale variation (dirt clumps)
    float mediumNoise = noise3D(pos.x * 1.5f, pos.y * 1.5f, pos.z * 1.5f);
    mediumNoise = mediumNoise * 0.3f + 0.7f; // Reduce contrast
    
    // Fine detail (grains)
    float grainNoise = noise3D(pos.x * 8.0f, pos.y * 8.0f, pos.z * 8.0f);
    grainNoise = grainNoise * 0.15f + 0.85f;
    
    // Add pebbles/stones occasionally
    float pebbleNoise = noise3D(pos.x * 4.0f + 100.0f, pos.y * 4.0f, pos.z * 4.0f);
    if (pebbleNoise > 0.8f) {
        // Make some spots look like small stones
        float stoneLevel = (pebbleNoise - 0.8f) * 5.0f; // 0 to 1
        Vec3 stoneColor(0.45f, 0.42f, 0.4f);
        soilColor = soilColor * (1.0f - stoneLevel * 0.5f) + stoneColor * (stoneLevel * 0.5f);
        grainNoise = lerp(grainNoise, 1.0f, stoneLevel * 0.3f);
    }
    
    // Combine all layers
    Vec3 finalColor = soilColor * mediumNoise * grainNoise;
    
    // Add slight normal-based shading for cracks
    float normalInfluence = std::abs(normal.y);
    finalColor = finalColor * (0.9f + normalInfluence * 0.1f);
    
    return finalColor;
}

// Procedural grass texture
inline Vec3 getGrassTexture(const Vec3& pos, const Vec3& normal) {
    // Base grass color
    Vec3 baseGreen(0.2f, 0.6f, 0.2f);
    
    // Create grass blade pattern
    float bladePattern = std::sin(pos.x * 20.0f) * std::cos(pos.z * 20.0f);
    bladePattern = bladePattern * 0.5f + 0.5f; // Normalize to 0-1
    
    // Large scale variation (grass patches)
    float patchNoise = fbm(pos.x * 0.15f, pos.y * 0.15f, pos.z * 0.15f, 3);
    Vec3 dryGrass(0.35f, 0.45f, 0.15f);  // Yellower grass
    Vec3 lushGrass(0.15f, 0.65f, 0.18f); // Deeper green
    Vec3 grassColor = dryGrass * (1.0f - patchNoise) + lushGrass * patchNoise;
    
    // Medium scale variation (individual grass clumps)
    float clumpNoise = noise3D(pos.x * 2.0f, pos.y * 2.0f, pos.z * 2.0f);
    clumpNoise = clumpNoise * 0.4f + 0.6f;
    
    // Fine grass blade detail
    float bladeDetail = noise3D(pos.x * 15.0f, pos.y * 15.0f, pos.z * 15.0f);
    bladeDetail = bladeDetail * 0.25f + 0.75f;
    
    // Add some brown patches (dead grass)
    float deadGrassNoise = noise3D(pos.x * 0.8f + 50.0f, pos.y * 0.8f, pos.z * 0.8f);
    if (deadGrassNoise > 0.75f) {
        float deadAmount = (deadGrassNoise - 0.75f) * 4.0f;
        Vec3 brownGrass(0.4f, 0.35f, 0.2f);
        grassColor = grassColor * (1.0f - deadAmount * 0.7f) + brownGrass * (deadAmount * 0.7f);
    }
    
    // Add occasional flowers/dandelions
    float flowerNoise = hash(std::floor(pos.x * 3.0f), 0.0f, std::floor(pos.z * 3.0f));
    if (flowerNoise > 0.95f) {
        float flowerCenterX = std::floor(pos.x * 3.0f) / 3.0f + 0.167f;
        float flowerCenterZ = std::floor(pos.z * 3.0f) / 3.0f + 0.167f;
        float distToFlower = std::sqrt((pos.x - flowerCenterX) * (pos.x - flowerCenterX) + 
                                       (pos.z - flowerCenterZ) * (pos.z - flowerCenterZ));
        if (distToFlower < 0.1f) {
            float flowerIntensity = 1.0f - (distToFlower / 0.1f);
            Vec3 flowerColor;
            if (flowerNoise > 0.98f) {
                flowerColor = Vec3(1.0f, 1.0f, 0.2f); // Yellow dandelion
            } else if (flowerNoise > 0.97f) {
                flowerColor = Vec3(1.0f, 0.7f, 0.8f); // Pink flower
            } else {
                flowerColor = Vec3(0.9f, 0.9f, 1.0f); // White flower
            }
            grassColor = grassColor * (1.0f - flowerIntensity * 0.8f) + flowerColor * (flowerIntensity * 0.8f);
        }
    }
    
    // Combine all layers with blade pattern
    Vec3 finalColor = grassColor * clumpNoise * bladeDetail;
    finalColor = finalColor * (0.8f + bladePattern * 0.2f);
    
    // Normal-based variation (slopes have different grass)
    float slope = 1.0f - std::abs(normal.y);
    if (slope > 0.3f) {
        // Steeper slopes have sparser, drier grass
        Vec3 slopeGrass(0.35f, 0.4f, 0.2f);
        finalColor = finalColor * (1.0f - slope * 0.3f) + slopeGrass * (slope * 0.3f);
    }
    
    return finalColor;
}

// Procedural sand texture (bonus, since you have sand blocks)
inline Vec3 getSandTexture(const Vec3& pos, const Vec3& normal) {
    // Base sand color
    Vec3 baseColor(0.76f, 0.7f, 0.5f);
    
    // Large scale dune pattern
    float dunePattern = fbm(pos.x * 0.3f, pos.y * 0.5f, pos.z * 0.3f, 2);
    Vec3 lightSand(0.85f, 0.8f, 0.65f);
    Vec3 darkSand(0.65f, 0.58f, 0.4f);
    Vec3 sandColor = darkSand * (1.0f - dunePattern) + lightSand * dunePattern;
    
    // Ripple pattern
    float ripples = std::sin(pos.x * 8.0f + pos.z * 5.0f) * 0.5f + 0.5f;
    ripples = ripples * 0.1f + 0.9f;
    
    // Fine sand grains
    float grainNoise = noise3D(pos.x * 20.0f, pos.y * 20.0f, pos.z * 20.0f);
    grainNoise = grainNoise * 0.08f + 0.92f;
    
    // Occasional shells or pebbles
    float debrisNoise = hash(std::floor(pos.x * 5.0f), std::floor(pos.y * 5.0f), std::floor(pos.z * 5.0f));
    if (debrisNoise > 0.92f) {
        float debrisIntensity = (debrisNoise - 0.92f) * 12.5f;
        Vec3 debrisColor = debrisNoise > 0.96f ? Vec3(0.9f, 0.9f, 0.85f) : Vec3(0.4f, 0.35f, 0.3f);
        sandColor = sandColor * (1.0f - debrisIntensity * 0.3f) + debrisColor * (debrisIntensity * 0.3f);
    }
    
    return sandColor * ripples * grainNoise;
}

// Procedural stone texture
inline Vec3 getStoneTexture(const Vec3& pos, const Vec3& normal) {
    // Base stone color
    Vec3 baseColor(0.5f, 0.5f, 0.5f);
    
    // Large scale variation (different stone types)
    float typeNoise = fbm(pos.x * 0.1f, pos.y * 0.1f, pos.z * 0.1f, 2);
    Vec3 graniteColor(0.55f, 0.5f, 0.48f);
    Vec3 basaltColor(0.3f, 0.32f, 0.35f);
    Vec3 stoneColor = basaltColor * (1.0f - typeNoise) + graniteColor * typeNoise;
    
    // Cracks and veins
    float crackNoise = fbm(pos.x * 2.0f, pos.y * 2.0f, pos.z * 2.0f, 4);
    if (crackNoise > 0.7f || crackNoise < 0.3f) {
        float crackIntensity = crackNoise > 0.7f ? (crackNoise - 0.7f) * 3.3f : (0.3f - crackNoise) * 3.3f;
        stoneColor = stoneColor * (1.0f - crackIntensity * 0.4f);
    }
    
    // Surface roughness
    float roughness = noise3D(pos.x * 10.0f, pos.y * 10.0f, pos.z * 10.0f);
    roughness = roughness * 0.2f + 0.8f;
    
    return stoneColor * roughness;
}

// Voxel World
class World {
    std::vector<uint8_t> blocks;
    int seed;
    
    float getTerrainHeight(int x, int z) const {
        float height = 12;
        height += 8 * std::sin(x * 0.05f + seed * 0.1f) * std::cos(z * 0.05f);
        height += 4 * std::sin(x * 0.1f) * std::cos(z * 0.15f + seed * 0.2f);
        height += 2 * std::sin(x * 0.3f + seed * 0.3f) * std::cos(z * 0.3f);
        return height;
    }
    
    void generateWaterBodies(int waterLevel) {
        std::queue<Vec3i> waterQueue;
        std::set<std::tuple<int,int,int>> visited;
        
        for (int x = 0; x < WORLD_SIZE; x++) {
            for (int z = 0; z < WORLD_SIZE; z++) {
                int terrainHeight = static_cast<int>(getTerrainHeight(x, z));
                terrainHeight = std::max(1, std::min(terrainHeight, WORLD_HEIGHT - 8));
                
                if (terrainHeight < waterLevel) {
                    for (int y = terrainHeight; y < waterLevel && y < WORLD_HEIGHT; y++) {
                        waterQueue.push(Vec3i(x, y, z));
                    }
                }
            }
        }
        
        std::vector<Vec3i> directions = {
            Vec3i(1, 0, 0), Vec3i(-1, 0, 0),
            Vec3i(0, 0, 1), Vec3i(0, 0, -1),
            Vec3i(0, -1, 0)
        };
        
        while (!waterQueue.empty()) {
            Vec3i pos = waterQueue.front();
            waterQueue.pop();
            
            auto key = std::make_tuple(pos.x, pos.y, pos.z);
            if (visited.count(key)) continue;
            visited.insert(key);
            
            if (pos.x < 0 || pos.x >= WORLD_SIZE || 
                pos.y < 0 || pos.y >= WORLD_HEIGHT || 
                pos.z < 0 || pos.z >= WORLD_SIZE) continue;
            
            if (getBlock(pos.x, pos.y, pos.z) == AIR) {
                setBlock(pos.x, pos.y, pos.z, WATER);
                
                for (const auto& dir : directions) {
                    Vec3i next = pos + dir;
                    if (next.y < waterLevel || dir.y < 0) {
                        waterQueue.push(next);
                    }
                }
            }
        }
        
        // Replace dirt/grass next to water with sand
        for (int x = 0; x < WORLD_SIZE; x++) {
            for (int z = 0; z < WORLD_SIZE; z++) {
                for (int y = 0; y < WORLD_HEIGHT; y++) {
                    if (getBlock(x, y, z) == WATER) {
                        for (int dx = -1; dx <= 1; dx++) {
                            for (int dz = -1; dz <= 1; dz++) {
                                for (int dy = -1; dy <= 0; dy++) {
                                    int nx = x + dx, ny = y + dy, nz = z + dz;
                                    if (nx >= 0 && nx < WORLD_SIZE && 
                                        ny >= 0 && ny < WORLD_HEIGHT && 
                                        nz >= 0 && nz < WORLD_SIZE) {
                                        BlockType block = getBlock(nx, ny, nz);
                                        if (block == DIRT || block == GRASS) {
                                            setBlock(nx, ny, nz, SAND);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
public:
    World() : blocks(WORLD_SIZE * WORLD_HEIGHT * WORLD_SIZE, AIR), seed(42) {}
    
    void generate(int newSeed) {
        seed = newSeed;
        std::fill(blocks.begin(), blocks.end(), AIR);
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        
        // Generate terrain
        for (int x = 0; x < WORLD_SIZE; x++) {
            for (int z = 0; z < WORLD_SIZE; z++) {
                int h = static_cast<int>(getTerrainHeight(x, z));
                h = std::max(1, std::min(h, WORLD_HEIGHT - 8));
                
                for (int y = 0; y < h && y < WORLD_HEIGHT; y++) {
                    BlockType type = STONE;
                    if (y == h - 1) type = GRASS;
                    else if (y >= h - 3) type = DIRT;
                    
                    setBlock(x, y, z, type);
                }
                
                // Trees
                if (dist(rng) < 0.03f && h > 12 && h < WORLD_HEIGHT - 8) {
                    int treeHeight = 4 + static_cast<int>(dist(rng) * 4);
                    for (int y = h; y < h + treeHeight; y++) {
                        setBlock(x, y, z, WOOD);
                    }
                    
                    int leafRadius = 2 + (dist(rng) > 0.5f ? 1 : 0);
                    for (int dx = -leafRadius; dx <= leafRadius; dx++) {
                        for (int dy = treeHeight - 2; dy <= treeHeight + 2; dy++) {
                            for (int dz = -leafRadius; dz <= leafRadius; dz++) {
                                if (std::abs(dx) + std::abs(dz) <= leafRadius + 1) {
                                    int nx = x + dx, ny = h + dy, nz = z + dz;
                                    if (nx >= 0 && nx < WORLD_SIZE && 
                                        nz >= 0 && nz < WORLD_SIZE && 
                                        ny < WORLD_HEIGHT) {
                                        if (getBlock(nx, ny, nz) == AIR && dist(rng) > 0.2f) {
                                            setBlock(nx, ny, nz, LEAVES);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                
                // Lights
                if (dist(rng) < 0.008f && h > 15 && h < WORLD_HEIGHT - 1) {
                    setBlock(x, h, z, LIGHT);
                }
            }
        }
        
        generateWaterBodies(11);
    }
    
    inline BlockType getBlock(int x, int y, int z) const {
        if (x < 0 || x >= WORLD_SIZE || y < 0 || y >= WORLD_HEIGHT || z < 0 || z >= WORLD_SIZE)
            return AIR;
        return static_cast<BlockType>(blocks[x + y * WORLD_SIZE + z * WORLD_SIZE * WORLD_HEIGHT]);
    }
    
    void setBlock(int x, int y, int z, BlockType type) {
        if (x >= 0 && x < WORLD_SIZE && y >= 0 && y < WORLD_HEIGHT && z >= 0 && z < WORLD_SIZE) {
            blocks[x + y * WORLD_SIZE + z * WORLD_SIZE * WORLD_HEIGHT] = type;
        }
    }
    
    bool raycast(const Ray& ray, float maxDist, Vec3& hitPos, Vec3& hitNormal, BlockType& hitBlock) const {
        Vec3 pos = ray.origin;
        Vec3 dir = ray.direction;
        
        int x = static_cast<int>(std::floor(pos.x));
        int y = static_cast<int>(std::floor(pos.y));
        int z = static_cast<int>(std::floor(pos.z));
        
        int stepX = dir.x > 0 ? 1 : -1;
        int stepY = dir.y > 0 ? 1 : -1;
        int stepZ = dir.z > 0 ? 1 : -1;
        
        float tMaxX = (dir.x != 0) ? ((x + (stepX > 0 ? 1 : 0)) - pos.x) / dir.x : 1e30f;
        float tMaxY = (dir.y != 0) ? ((y + (stepY > 0 ? 1 : 0)) - pos.y) / dir.y : 1e30f;
        float tMaxZ = (dir.z != 0) ? ((z + (stepZ > 0 ? 1 : 0)) - pos.z) / dir.z : 1e30f;
        
        float tDeltaX = (dir.x != 0) ? stepX / dir.x : 1e30f;
        float tDeltaY = (dir.y != 0) ? stepY / dir.y : 1e30f;
        float tDeltaZ = (dir.z != 0) ? stepZ / dir.z : 1e30f;
        
        float dist = 0;
        Vec3 normal(0, 1, 0);
        
        BlockType startBlock = getBlock(static_cast<int>(std::floor(ray.origin.x)),
                                        static_cast<int>(std::floor(ray.origin.y)),
                                        static_cast<int>(std::floor(ray.origin.z)));
        bool startedInWater = (startBlock == WATER);
        bool currentlyInWater = startedInWater;
        
        while (dist < maxDist) {
            BlockType block = getBlock(x, y, z);
            
            if (currentlyInWater && block == AIR) {
                hitPos = pos + dir * dist;
                hitBlock = WATER;
                hitNormal = normal;
                return true;
            } else if (!currentlyInWater && block == WATER) {
                hitPos = pos + dir * dist;
                hitBlock = WATER;
                hitNormal = normal;
                return true;
            }
            
            if (block != AIR && block != WATER) {
                hitPos = pos + dir * dist;
                hitBlock = block;
                hitNormal = normal;
                return true;
            }
            
            if (block == WATER) {
                currentlyInWater = true;
            } else if (block == AIR) {
                currentlyInWater = false;
            }
            
            if (tMaxX < tMaxY) {
                if (tMaxX < tMaxZ) {
                    x += stepX;
                    dist = tMaxX;
                    tMaxX += tDeltaX;
                    normal = Vec3(-stepX, 0, 0);
                } else {
                    z += stepZ;
                    dist = tMaxZ;
                    tMaxZ += tDeltaZ;
                    normal = Vec3(0, 0, -stepZ);
                }
            } else {
                if (tMaxY < tMaxZ) {
                    y += stepY;
                    dist = tMaxY;
                    tMaxY += tDeltaY;
                    normal = Vec3(0, -stepY, 0);
                } else {
                    z += stepZ;
                    dist = tMaxZ;
                    tMaxZ += tDeltaZ;
                    normal = Vec3(0, 0, -stepZ);
                }
            }
            
            if (x < 0 || x >= WORLD_SIZE || y < 0 || y >= WORLD_HEIGHT || z < 0 || z >= WORLD_SIZE) {
                break;
            }
        }
        
        return false;
    }
};

// Camera
class Camera {
public:
    Vec3 position;
    float yaw, pitch;
    
    Camera() : position(WORLD_SIZE/2, 30, WORLD_SIZE/2), yaw(0), pitch(0) {}
    
    Vec3 getForward() const {
        return Vec3(
            std::sin(yaw) * std::cos(pitch),
            std::sin(pitch),
            std::cos(yaw) * std::cos(pitch)
        ).normalize();
    }
    
    Vec3 getRight() const {
        return Vec3(std::sin(yaw - M_PI/2), 0, std::cos(yaw - M_PI/2)).normalize();
    }
    
    Vec3 getUp() const {
        return getRight().cross(getForward()).normalize();
    }
    
    Ray getRay(float u, float v, float aspectRatio) const {
        float fovRad = FOV * M_PI / 180.0f;
        float halfHeight = std::tan(fovRad / 2);
        float halfWidth = aspectRatio * halfHeight;
        
        Vec3 forward = getForward();
        Vec3 right = getRight();
        Vec3 up = getUp();
        
        Vec3 direction = forward + right * (u * halfWidth) + up * (v * halfHeight);
        return Ray(position, direction.normalize());
    }
    
    void setFromKeyframe(float x, float y, float z, float yaw, float pitch) {
        position.x = x;
        position.y = y;
        position.z = z;
        this->yaw = yaw;
        this->pitch = pitch;
    }
};

// Enhanced water surface normal for better caustics
inline Vec3 getWaterNormal(const Vec3& pos, float time) {
    // Multi-frequency waves for complex caustic patterns
    float wave1 = std::sin(pos.x * 2.0f + time * 1.2f) * std::cos(pos.z * 1.8f - time * 0.9f);
    float wave2 = std::sin(pos.x * 3.5f - time * 1.5f) * std::cos(pos.z * 3.2f + time * 1.1f) * 0.5f;
    float wave3 = std::sin((pos.x + pos.z) * 1.2f + time * 0.7f) * 0.3f;
    float wave4 = std::sin(pos.x * 5.0f + time * 2.0f) * std::cos(pos.z * 4.5f - time * 1.8f) * 0.25f;
    
    float height = (wave1 + wave2 + wave3 + wave4) * 0.12f;
    
    // Calculate derivatives for normal
    float dx = 0.12f * (
        2.0f * std::cos(pos.x * 2.0f + time * 1.2f) * std::cos(pos.z * 1.8f - time * 0.9f) +
        3.5f * std::cos(pos.x * 3.5f - time * 1.5f) * std::cos(pos.z * 3.2f + time * 1.1f) * 0.5f +
        1.2f * std::cos((pos.x + pos.z) * 1.2f + time * 0.7f) * 0.3f +
        5.0f * std::cos(pos.x * 5.0f + time * 2.0f) * std::cos(pos.z * 4.5f - time * 1.8f) * 0.25f
    );
    
    float dz = 0.12f * (
        -1.8f * std::sin(pos.x * 2.0f + time * 1.2f) * std::sin(pos.z * 1.8f - time * 0.9f) +
        3.2f * std::sin(pos.x * 3.5f - time * 1.5f) * std::sin(pos.z * 3.2f + time * 1.1f) * 0.5f +
        1.2f * std::cos((pos.x + pos.z) * 1.2f + time * 0.7f) * 0.3f -
        4.5f * std::sin(pos.x * 5.0f + time * 2.0f) * std::sin(pos.z * 4.5f - time * 1.8f) * 0.25f
    );
    
    return Vec3(-dx, 1.0f, -dz).normalize();
}

// Get sky color
Vec3 getSkyColor(const Vec3& direction, float timeOfDay, const SunLight& sun) {
    float y = direction.y;
    float t = 0.5f * (y + 1.0f);
    
    Vec3 horizonColor, zenithColor;
    
    if (timeOfDay < 0.25f) {
        float dawn = timeOfDay * 4.0f;
        horizonColor = Vec3(1.0f, 0.6f, 0.3f) * dawn + Vec3(0.1f, 0.1f, 0.2f) * (1 - dawn);
        zenithColor = Vec3(0.3f, 0.4f, 0.8f) * dawn + Vec3(0.05f, 0.05f, 0.1f) * (1 - dawn);
    } else if (timeOfDay < 0.75f) {
        horizonColor = Vec3(0.5f, 0.7f, 1.0f);
        zenithColor = Vec3(0.2f, 0.4f, 0.8f);
    } else {
        float dusk = (timeOfDay - 0.75f) * 4.0f;
        horizonColor = Vec3(0.5f, 0.7f, 1.0f) * (1 - dusk) + Vec3(1.0f, 0.5f, 0.3f) * dusk;
        zenithColor = Vec3(0.2f, 0.4f, 0.8f) * (1 - dusk) + Vec3(0.2f, 0.1f, 0.3f) * dusk;
    }
    
    Vec3 skyGradient = horizonColor * (1 - t) + zenithColor * t;
    
    float sunDot = direction.dot(sun.direction);
    if (sunDot < -0.999f) {
        float sunGlow = std::pow((-sunDot - 0.999f) * 1000.0f, 2.0f);
        Vec3 sunColor = sun.color * 5.0f;
        skyGradient = skyGradient + sunColor * sunGlow;
    } else if (sunDot < -0.99f) {
        float glow = std::pow((-sunDot - 0.99f) * 100.0f, 0.5f);
        skyGradient = skyGradient + sun.color * glow * 0.5f;
    }
    
    return skyGradient;
}

// Calculate volumetrics
Vec3 calculateVolumetrics(const Ray& ray, float maxDist, const World& world, const SunLight& sun, bool inWater) {
    Vec3 volumetricLight(0, 0, 0);
    
    if (!g_settings.enableVolumetrics) return volumetricLight;
    
    const int numSamples = 12;
    float stepSize = std::min(maxDist, 50.0f) / float(numSamples);
    
    float scatteringCoeff = inWater ? 0.2f : 0.04f;
    float absorptionCoeff = inWater ? 0.08f : 0.01f;
    
    float cosTheta = ray.direction.dot(-sun.direction);
    float g = inWater ? 0.8f : 0.6f;
    float phase = (1.0f - g * g) / (4.0f * M_PI * std::pow(1.0f + g * g - 2.0f * g * cosTheta, 1.5f));
    
    for (int i = 0; i < numSamples; i++) {
        float t = stepSize * (i + random01() * 0.5f);
        Vec3 samplePos = ray.at(t);
        
        int sx = static_cast<int>(std::floor(samplePos.x));
        int sy = static_cast<int>(std::floor(samplePos.y));
        int sz = static_cast<int>(std::floor(samplePos.z));
        
        bool sampleInWater = (world.getBlock(sx, sy, sz) == WATER);
        
        if (sampleInWater != inWater) continue;
        
        Ray shadowRay(samplePos, -sun.direction);
        Vec3 shadowHit, shadowNormal;
        BlockType shadowBlock;
        
        bool sunVisible = !world.raycast(shadowRay, 100.0f, shadowHit, shadowNormal, shadowBlock);
        float intensity = sun.intensity;
        
        if (shadowBlock == WATER && !inWater) {
            sunVisible = true;
            intensity *= 0.5f;
        } else if (shadowBlock == LEAVES) {
            sunVisible = true;
            intensity *= 0.3f;
        }
        
        if (sunVisible) {
            if (inWater) {
                float depth = std::max(0.0f, 11.0f - samplePos.y);
                intensity *= std::exp(-depth * 0.05f);
            }
            
            if (!inWater) {
                float heightFactor = std::exp(-(samplePos.y - 10.0f) * 0.02f);
                heightFactor = std::max(0.1f, std::min(1.0f, heightFactor));
                intensity *= heightFactor;
            }
            
            Vec3 inScattering = sun.color * intensity * scatteringCoeff * phase;
            float absorption = std::exp(-t * absorptionCoeff);
            volumetricLight = volumetricLight + inScattering * absorption * stepSize;
        }
    }
    
    if (inWater) {
        volumetricLight = volumetricLight * Vec3(0.7f, 0.9f, 1.0f);
    } else {
        float timeStrength = 1.0f + 2.0f * (1.0f - std::abs(g_settings.timeOfDay - 0.5f) * 2.0f);
        volumetricLight = volumetricLight * Vec3(1.0f, 0.95f, 0.9f) * timeStrength;
    }
    
    return volumetricLight * 3.0f;
}

// Physically-based caustics calculation
float calculateCaustics(const Vec3& pos, const World& world, const SunLight& sun, float time) {
    if (!g_settings.enableCaustics) return 0.0f;
    
    // Early exit if above water
    if (pos.y > 11.0f) return 0.0f;
    
    // Check if position is underwater
    int checkX = static_cast<int>(std::floor(pos.x));
    int checkY = static_cast<int>(std::floor(pos.y));
    int checkZ = static_cast<int>(std::floor(pos.z));
    
    bool underWater = false;
    for (int y = checkY + 1; y <= 11 && y < WORLD_HEIGHT; y++) {
        if (world.getBlock(checkX, y, checkZ) == WATER) {
            underWater = true;
            break;
        }
    }
    
    if (!underWater) return 0.0f;
    
    float causticIntensity = 0.0f;
    
    // Number of samples based on quality setting
    int samples = 8;  // Default low quality
    switch(g_settings.causticQuality) {
        case 1: samples = 8; break;   // Low - fast
        case 2: samples = 16; break;  // Medium - balanced
        case 3: samples = 32; break;  // High - for offline rendering
    }
    
    // For offline rendering, always use high quality
    if (g_settings.mode == Settings::MODE_OFFLINE_RENDER) {
        samples = 32;
    }
    
    // Sample area size - smaller = sharper caustics
    float sampleRadius = 1.5f;
    
    for (int i = 0; i < samples; i++) {
        // Sample point on water surface above (stratified sampling for better quality)
        float angle = (i + random01()) * 2.0f * M_PI / float(samples);
        float radius = std::sqrt(random01()) * sampleRadius;
        float offsetX = radius * std::cos(angle);
        float offsetZ = radius * std::sin(angle);
        
        Vec3 waterSurfacePos(pos.x + offsetX, 11.0f, pos.z + offsetZ);
        
        // Get water surface normal at this point
        Vec3 waterNormal = getWaterNormal(waterSurfacePos, time);
        
        // Calculate refracted ray direction through Snell's law
        Vec3 incident = sun.direction;
        float n1 = 1.0f;    // air
        float n2 = 1.333f;  // water
        float ratio = n1 / n2;
        
        float cosI = -waterNormal.dot(incident);
        
        // Handle edge case where ray grazes surface
        if (cosI < 0.01f) continue;
        
        float sinT2 = ratio * ratio * (1.0f - cosI * cosI);
        
        if (sinT2 <= 1.0f) {
            float cosT = std::sqrt(1.0f - sinT2);
            Vec3 refracted = incident * ratio + waterNormal * (ratio * cosI - cosT);
            refracted = refracted.normalize();
            
            // Check if refracted ray hits near our position
            if (refracted.y < -0.01f) {  // Ray must be going down
                float t = (pos.y - waterSurfacePos.y) / refracted.y;
                Vec3 hitPos = waterSurfacePos + refracted * t;
                
                float distance = std::sqrt((hitPos.x - pos.x) * (hitPos.x - pos.x) + 
                                          (hitPos.z - pos.z) * (hitPos.z - pos.z));
                
                // Gaussian falloff for smooth caustics
                float contribution = std::exp(-distance * distance * 4.0f);
                
                // Add focusing factor based on normal deviation
                float focusFactor = 1.0f + (1.0f - std::abs(waterNormal.y)) * 2.0f;
                contribution *= focusFactor;
                
                causticIntensity += contribution;
            }
        }
    }
    
    // Normalize by sample count
    causticIntensity /= float(samples);
    
    // Artistic adjustments
    causticIntensity *= 12.0f;  // Overall brightness
    
    // Add sharp highlights
    causticIntensity = std::pow(causticIntensity, 1.5f) * 1.5f;
    
    // Depth attenuation
    float depth = 11.0f - pos.y;
    float depthFade = std::exp(-depth * 0.02f);
    
    // Add subtle color variation based on intensity (chromatic aberration effect)
    // This is applied in the main trace function
    
    return causticIntensity * depthFade * sun.intensity;
}

// Path tracing (simplified for space, same as original)
Vec3 trace(const Ray& ray, const World& world, int depth, bool insideWater = false);

// Renderer - modified to support offline rendering
class Renderer {
    std::vector<uint32_t> framebuffer;
    std::vector<Vec3> accumulator;
    std::atomic<int> nextTile;
    int sampleCount;
    int currentWidth, currentHeight;
    static constexpr int TILE_SIZE = 8;
    
    bool getCameraUnderwater(const Camera& camera, const World& world) const {
        int camX = static_cast<int>(std::floor(camera.position.x));
        int camY = static_cast<int>(std::floor(camera.position.y));
        int camZ = static_cast<int>(std::floor(camera.position.z));
        return (world.getBlock(camX, camY, camZ) == WATER);
    }
    
public:
    Renderer() : nextTile(0), sampleCount(0), currentWidth(0), currentHeight(0) {
        resize(g_settings.renderWidth, g_settings.renderHeight);
    }
    
    void resize(int width, int height) {
        if (width != currentWidth || height != currentHeight) {
            currentWidth = width;
            currentHeight = height;
            framebuffer.resize(width * height);
            accumulator.resize(width * height);
            reset();
        }
    }
    
    void reset() {
        sampleCount = 0;
        std::fill(accumulator.begin(), accumulator.end(), Vec3(0, 0, 0));
    }
    
    void render(const Camera& camera, const World& world, bool cameraMoving) {
        if (cameraMoving) {
            reset();
        }
        
        g_settings.waterAnimation += 0.05f;
        
        nextTile = 0;
        sampleCount++;
        
        int numThreads = std::thread::hardware_concurrency();
        std::vector<std::thread> threads;
        
        bool cameraUnderwater = getCameraUnderwater(camera, world);
        
        for (int i = 0; i < numThreads; i++) {
            threads.emplace_back([this, &camera, &world, cameraUnderwater]() {
                renderThread(camera, world, cameraUnderwater);
            });
        }
        
        for (auto& t : threads) {
            t.join();
        }
        
        // Convert accumulator to framebuffer
        for (int i = 0; i < currentWidth * currentHeight; i++) {
            Vec3 color = accumulator[i] / float(sampleCount * SAMPLES_PER_PIXEL);
            
            color.x = color.x / (1.0f + color.x);
            color.y = color.y / (1.0f + color.y);
            color.z = color.z / (1.0f + color.z);
            
            color.x = std::pow(color.x, 1.0f / 2.2f);
            color.y = std::pow(color.y, 1.0f / 2.2f);
            color.z = std::pow(color.z, 1.0f / 2.2f);
            
            uint8_t r = static_cast<uint8_t>(std::min(color.x * 255.0f, 255.0f));
            uint8_t g = static_cast<uint8_t>(std::min(color.y * 255.0f, 255.0f));
            uint8_t b = static_cast<uint8_t>(std::min(color.z * 255.0f, 255.0f));
            
            framebuffer[i] = (r << 16) | (g << 8) | b;
        }
    }
    
    void renderThread(const Camera& camera, const World& world, bool cameraUnderwater) {
        float aspectRatio = float(currentWidth) / currentHeight;
        int tilesX = (currentWidth + TILE_SIZE - 1) / TILE_SIZE;
        int tilesY = (currentHeight + TILE_SIZE - 1) / TILE_SIZE;
        int totalTiles = tilesX * tilesY;
        
        while (true) {
            int tileIndex = nextTile.fetch_add(1);
            if (tileIndex >= totalTiles) break;
            
            int tileX = tileIndex % tilesX;
            int tileY = tileIndex / tilesX;
            int startX = tileX * TILE_SIZE;
            int startY = tileY * TILE_SIZE;
            int endX = std::min(startX + TILE_SIZE, currentWidth);
            int endY = std::min(startY + TILE_SIZE, currentHeight);
            
            for (int y = startY; y < endY; y++) {
                for (int x = startX; x < endX; x++) {
                    Vec3 color(0, 0, 0);
                    
                    for (int s = 0; s < SAMPLES_PER_PIXEL; s++) {
                        float u = (x + random01() - currentWidth/2.0f) / (currentWidth/2.0f);
                        float v = -(y + random01() - currentHeight/2.0f) / (currentHeight/2.0f);
                        
                        Ray ray = camera.getRay(u, v, aspectRatio);
                        color = color + trace(ray, world, MAX_BOUNCES, cameraUnderwater);
                    }
                    
                    int idx = y * currentWidth + x;
                    accumulator[idx] = accumulator[idx] + color;
                }
            }
        }
    }
    
    const uint32_t* getFramebuffer() const { return framebuffer.data(); }
    int getSampleCount() const { return sampleCount * SAMPLES_PER_PIXEL; }
    int getWidth() const { return currentWidth; }
    int getHeight() const { return currentHeight; }
    
    void saveFrame(const std::string& filename) {
        // Save as PPM for simplicity
        std::ofstream file(filename);
        file << "P3\n" << currentWidth << " " << currentHeight << "\n255\n";
        
        for (int i = 0; i < currentWidth * currentHeight; i++) {
            uint32_t pixel = framebuffer[i];
            int r = (pixel >> 16) & 0xFF;
            int g = (pixel >> 8) & 0xFF;
            int b = pixel & 0xFF;
            file << r << " " << g << " " << b << "\n";
        }
        
        file.close();
    }
};

// Complete trace function implementation
Vec3 trace(const Ray& ray, const World& world, int depth, bool insideWater) {
    if (depth <= 0) return Vec3(0, 0, 0);
    
    SunLight sun;
    sun.updateFromTimeOfDay(g_settings.timeOfDay);
    
    Vec3 hitPos, hitNormal;
    BlockType hitBlock;
    
    // Calculate distance to hit (or max distance if no hit)
    float hitDistance = MAX_RAY_DISTANCE;
    bool didHit = world.raycast(ray, MAX_RAY_DISTANCE, hitPos, hitNormal, hitBlock);
    if (didHit) {
        hitDistance = (hitPos - ray.origin).length();
    }
    
    // Calculate volumetric lighting along the ray
    Vec3 volumetrics(0, 0, 0);
    if (g_settings.enableVolumetrics) {
        volumetrics = calculateVolumetrics(ray, hitDistance, world, sun, insideWater);
    }
    
    if (!didHit) {
        if (insideWater) {
            float depth = std::max(0.0f, 11.0f - ray.origin.y);
            float depthFactor = std::exp(-depth * 0.08f);
            // Tropical blue-green underwater ambience
            Vec3 baseColor = Vec3(0.02f, 0.08f, 0.12f) * (0.4f + 0.6f * depthFactor);
            return baseColor + volumetrics;
        }
        
        Vec3 skyColor = getSkyColor(ray.direction, g_settings.timeOfDay, sun);
        return skyColor + volumetrics;
    }
    
    const MaterialProps& mat = g_materials[hitBlock];
    
    // Handle water
    if (hitBlock == WATER) {
        bool entering = !insideWater;
        Vec3 normal = entering ? hitNormal : hitNormal * -1;
        
        // Add waves on top surface
        if (entering && hitNormal.y > 0.9f) {
            Vec3 waveNormal = getWaterNormal(hitPos, g_settings.waterAnimation);
            normal = (normal * 0.8f + waveNormal * 0.2f).normalize();
        }
        
        float n1 = entering ? 1.0f : 1.333f;
        float n2 = entering ? 1.333f : 1.0f;
        float n = n1 / n2;
        
        float cosI = -normal.dot(ray.direction);
        float sinT2 = n * n * (1.0f - cosI * cosI);
        
        Vec3 refractedDir;
        bool totalInternalReflection = false;
        
        if (sinT2 <= 1.0f) {
            float cosT = std::sqrt(1.0f - sinT2);
            refractedDir = ray.direction * n + normal * (n * cosI - cosT);
        } else {
            totalInternalReflection = true;
            refractedDir = ray.direction - normal * 2.0f * ray.direction.dot(normal);
        }
        
        Vec3 offsetPos = hitPos + refractedDir * 0.01f;
        Ray refractedRay(offsetPos, refractedDir);
        
        bool rayNowInsideWater = entering && !totalInternalReflection;
        Vec3 transmitted = trace(refractedRay, world, depth - 1, rayNowInsideWater);
        
        // Water absorption with tropical blue tint
        if (entering && !totalInternalReflection) {
            // Tropical blue-teal water color
            Vec3 waterTint(0.05f, 0.25f, 0.35f);
            float distance = (hitPos - ray.origin).length();
            float absorption = std::exp(-distance * 0.08f);  // Stronger absorption for more color
            transmitted = transmitted * absorption + waterTint * (1.0f - absorption) * 0.4f;
        }
        
        // Fresnel reflectance for water surface
        float r0 = ((n1 - n2) / (n1 + n2)) * ((n1 - n2) / (n1 + n2));
        float reflectance = r0 + (1.0f - r0) * std::pow(1.0f - std::abs(cosI), 5.0f);
        
        // Add reflections only when looking at water from above
        if (!insideWater && !totalInternalReflection) {
            reflectance = std::min(reflectance, 0.8f);  // Reduced max reflectance for more water color
            
            if (reflectance > 0.02f) {
                Vec3 reflectedDir = ray.direction - normal * 2.0f * ray.direction.dot(normal);
                Ray reflectedRay(hitPos + normal * 0.01f, reflectedDir);
                Vec3 reflected = trace(reflectedRay, world, depth - 1, false);
                
                // Mix in water color even with reflections
                Vec3 waterColor(0.1f, 0.35f, 0.45f);
                transmitted = transmitted * (1.0f - reflectance) + reflected * reflectance;
                transmitted = transmitted * 0.9f + waterColor * 0.1f;  // Always show some water color
                return transmitted + volumetrics;  // Add volumetrics
            }
        }
        
        return transmitted + volumetrics;  // Add volumetrics
    }
    
    // Check if surface is underwater
    bool actuallyUnderwater = false;
    int checkX = static_cast<int>(std::floor(hitPos.x));
    int checkY = static_cast<int>(std::floor(hitPos.y + 0.5f));
    int checkZ = static_cast<int>(std::floor(hitPos.z));
    
    for (int y = checkY; y < WORLD_HEIGHT && y < checkY + 10; y++) {
        if (world.getBlock(checkX, y, checkZ) == WATER) {
            actuallyUnderwater = true;
            break;
        }
    }
    
    bool isUnderwater = insideWater || actuallyUnderwater;
    
    // Handle emissive materials
    Vec3 emission = mat.emission;
    if (hitBlock == LIGHT) {
        float brightness = 0.3f + 0.7f * std::abs(g_settings.timeOfDay - 0.5f) * 2.0f;
        emission = emission * brightness;
        if (isUnderwater) {
            float depth = std::max(0.0f, 11.0f - hitPos.y);
            emission = emission * (0.7f * std::exp(-depth * 0.03f));
        }
    }
    
    Vec3 color = emission;
    
    if (emission.x == 0 && emission.y == 0 && emission.z == 0) {
        // Get the base albedo from material properties
        Vec3 albedo = mat.albedo;
        
        // Apply procedural textures based on block type
        switch(hitBlock) {
            case DIRT:
                albedo = getDirtTexture(hitPos, hitNormal);
                break;
            case GRASS:
                albedo = getGrassTexture(hitPos, hitNormal);
                break;
            case SAND:
                albedo = getSandTexture(hitPos, hitNormal);
                break;
            case STONE:
                albedo = getStoneTexture(hitPos, hitNormal);
                break;
            default:
                // Use default material albedo for other block types
                break;
        }
        
        // Direct sun lighting
        Vec3 toSun = sun.direction * -1;
        Ray shadowRay(hitPos + hitNormal * 0.01f, toSun);
        Vec3 shadowHit, shadowNormal;
        BlockType shadowBlock;
        
        bool sunVisible = !world.raycast(shadowRay, 100.0f, shadowHit, shadowNormal, shadowBlock);
        if (shadowBlock == WATER) sunVisible = true;
        
        float sunDot = std::max(0.0f, hitNormal.dot(toSun));
        float sunStrength = sunDot * sun.intensity;
        
        if (isUnderwater) {
            float depth = std::max(0.0f, 11.0f - hitPos.y);
            float depthAttenuation = std::exp(-depth * 0.05f);
            sunStrength *= 0.3f * depthAttenuation;
        }
        
        Vec3 directLight = sunVisible ? sun.getLightContribution() * sunStrength : Vec3(0, 0, 0);
        
        // Add physically-based caustics
        if (isUnderwater) {
            float caustics = calculateCaustics(hitPos, world, sun, g_settings.waterAnimation);
            
            // Add chromatic aberration to caustics for realism
            Vec3 causticsColor;
            causticsColor.x = sun.color.x * (1.0f + caustics * 0.1f);
            causticsColor.y = sun.color.y;
            causticsColor.z = sun.color.z * (1.0f - caustics * 0.05f);
            
            directLight = directLight + causticsColor * caustics * 0.8f;
        }
        
        // Indirect lighting
        Vec3 target = hitPos + hitNormal + randomInHemisphere(hitNormal);
        Ray scattered(hitPos + hitNormal * 0.01f, (target - hitPos).normalize());
        
        float ambientStrength = isUnderwater ? 0.2f : (0.3f + 0.2f * sun.intensity);
        Vec3 indirectLight = trace(scattered, world, depth - 1, isUnderwater) * ambientStrength;
        
        // Ambient term
        Vec3 ambient = isUnderwater ? 
            Vec3(0.05f, 0.15f, 0.22f) * (0.3f + 0.7f * std::exp(-std::max(0.0f, 11.0f - hitPos.y) * 0.05f)) :
            Vec3(0.05f, 0.05f, 0.05f);
        
        // Use the procedurally textured albedo in the final color calculation
        color = color + albedo * (directLight + indirectLight + ambient);
    }
        
    // Underwater fog - tropical blue
    if (isUnderwater) {
        float distance = (hitPos - ray.origin).length();
        float fogFactor = std::exp(-distance * 0.025f);  // Thicker fog for more color
        Vec3 fogColor(0.04f, 0.18f, 0.28f);  // Tropical blue fog
        color = color * fogFactor + fogColor * (1.0f - fogFactor);
    }
    
    // Add volumetric lighting contribution
    return color + volumetrics;
}

// Main function with demo recording
int main(int argc, char* argv[]) {
    // Parse command line arguments
    bool offlineMode = false;
    bool benchmarkMode = false;
    std::string demoFile = "demo.json";
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--offline") {
            offlineMode = true;
            g_settings.mode = Settings::MODE_OFFLINE_RENDER;
        } else if (arg == "--benchmark") {
            benchmarkMode = true;
            g_settings.mode = Settings::MODE_BENCHMARK;
        } else if (arg == "--demo" && i + 1 < argc) {
            demoFile = argv[++i];
        } else if (arg == "--samples" && i + 1 < argc) {
            g_settings.offlineTargetSamples = std::stoi(argv[++i]);
        } else if (arg == "--resolution" && i + 1 < argc) {
            int preset = std::stoi(argv[++i]);
            g_settings.adjustRenderResolution(preset);
        } else if (arg == "--caustic-quality" && i + 1 < argc) {
            g_settings.causticQuality = std::stoi(argv[++i]);
        }
    }
    
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        std::cerr << "SDL init failed: " << SDL_GetError() << std::endl;
        return 1;
    }
    
    SDL_Window* window = nullptr;
    SDL_Renderer* sdlRenderer = nullptr;
    SDL_Texture* texture = nullptr;
    
    if (!offlineMode) {
        window = SDL_CreateWindow(
            "CPU Pathtracer [v5.0] - Physical Caustics", 
            SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
            g_settings.windowWidth, g_settings.windowHeight, 
            SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE
        );
        
        sdlRenderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
        texture = SDL_CreateTexture(
            sdlRenderer, SDL_PIXELFORMAT_RGB888,
            SDL_TEXTUREACCESS_STREAMING, 
            g_settings.renderWidth, g_settings.renderHeight
        );
        SDL_SetTextureScaleMode(texture, SDL_ScaleModeNearest);
        SDL_SetRelativeMouseMode(SDL_TRUE);
    }
    
    World world;
    world.generate(g_settings.worldSeed);
    
    Camera camera;
    Camera prevCamera = camera;
    Renderer renderer;
    
    DemoPath demoPath;
    BenchmarkRecorder benchmarkRecorder;
    
    bool running = true;
    const Uint8* keystate = SDL_GetKeyboardState(nullptr);
    
    auto startTime = std::chrono::high_resolution_clock::now();
    auto lastTime = startTime;
    auto lastRecordTime = startTime;
    auto lastFPSTime = startTime;  // Separate timer for FPS
    int frameCount = 0;
    int currentFPS = 0;
    float demoTime = 0;
    
    // Load demo if in playback/benchmark/offline mode
    if (g_settings.mode == Settings::MODE_PLAYBACK || 
        g_settings.mode == Settings::MODE_BENCHMARK || 
        g_settings.mode == Settings::MODE_OFFLINE_RENDER) {
        if (!demoPath.loadFromFile(demoFile)) {
            std::cerr << "Failed to load demo file: " << demoFile << "\n";
            return 1;
        }
    }
    
    // Setup benchmark recorder
    if (g_settings.mode == Settings::MODE_BENCHMARK) {
        benchmarkRecorder.systemInfo = SystemInfo::get();
        benchmarkRecorder.renderWidth = g_settings.renderWidth;
        benchmarkRecorder.renderHeight = g_settings.renderHeight;
    }
    
    std::cout << "\n=== CPU Pathtracer [v5.0] - Physical Caustics Edition ===\n";
    std::cout << "Mode: ";
    switch (g_settings.mode) {
        case Settings::MODE_INTERACTIVE: std::cout << "Interactive\n"; break;
        case Settings::MODE_RECORDING: std::cout << "Recording Demo\n"; break;
        case Settings::MODE_PLAYBACK: std::cout << "Playing Demo\n"; break;
        case Settings::MODE_BENCHMARK: std::cout << "Benchmark\n"; break;
        case Settings::MODE_OFFLINE_RENDER: std::cout << "Offline Render\n"; break;
    }
    
    std::cout << "\nControls:\n";
    std::cout << "F1: Start/Stop Recording | F2: Play Demo | F3: Benchmark\n";
    std::cout << "F5: Save Demo | F6: Load Demo\n";
    std::cout << "Movement: WASD + Space/Shift | Look: Mouse\n";
    std::cout << "Render Res: 1-6 | Window Size: Q/E\n";
    std::cout << "New World: R/F | Time: T/G | Quit: ESC\n";
    std::cout << "KP1: Toggle Caustics | KP2: Toggle Volumetrics\n";
    std::cout << "KP3: Caustic Quality (Low/Med/High)\n\n";
    
    // Create output directory for offline rendering
    if (g_settings.mode == Settings::MODE_OFFLINE_RENDER) {
#ifdef _WIN32
        system(("mkdir " + g_settings.outputDir).c_str());
#else
        system(("mkdir -p " + g_settings.outputDir).c_str());
#endif
    }
    
    int offlineFrameCount = 0;
    
    while (running) {
        auto currentTime = std::chrono::high_resolution_clock::now();
        float deltaTime = std::chrono::duration<float>(currentTime - lastTime).count();
        lastTime = currentTime;
        
        float totalElapsed = std::chrono::duration<float>(currentTime - startTime).count();
        
        bool cameraMoving = false;
        bool needsReset = false;
        
        if (!offlineMode) {
            SDL_Event event;
            while (SDL_PollEvent(&event)) {
                if (event.type == SDL_QUIT) {
                    running = false;
                } else if (event.type == SDL_KEYDOWN) {
                    switch(event.key.keysym.sym) {
                        case SDLK_ESCAPE:
                            running = false;
                            break;
                        
                        case SDLK_F1:
                            if (g_settings.mode == Settings::MODE_RECORDING) {
                                g_settings.mode = Settings::MODE_INTERACTIVE;
                                std::cout << "Recording stopped. " << demoPath.keyframes.size() << " keyframes recorded.\n";
                            } else {
                                g_settings.mode = Settings::MODE_RECORDING;
                                demoPath.clear();
                                startTime = currentTime;
                                std::cout << "Recording started...\n";
                            }
                            break;
                        
                        case SDLK_F2:
                            if (demoPath.keyframes.size() > 0) {
                                g_settings.mode = Settings::MODE_PLAYBACK;
                                demoTime = 0;
                                std::cout << "Playing demo...\n";
                            }
                            break;
                        
                        case SDLK_F3:
                            if (demoPath.keyframes.size() > 0) {
                                g_settings.mode = Settings::MODE_BENCHMARK;
                                demoTime = 0;
                                benchmarkRecorder.frames.clear();
                                benchmarkRecorder.systemInfo = SystemInfo::get();
                                benchmarkRecorder.renderWidth = g_settings.renderWidth;
                                benchmarkRecorder.renderHeight = g_settings.renderHeight;
                                std::cout << "Benchmark started...\n";
                            }
                            break;
                        
                        case SDLK_F5:
                            demoPath.saveToFile(demoFile);
                            break;
                        
                        case SDLK_F6:
                            demoPath.loadFromFile(demoFile);
                            break;
                        
                        // Other controls same as original
                        case SDLK_1: case SDLK_2: case SDLK_3:
                        case SDLK_4: case SDLK_5: case SDLK_6:
                            g_settings.adjustRenderResolution(event.key.keysym.sym - SDLK_0);
                            renderer.resize(g_settings.renderWidth, g_settings.renderHeight);
                            if (texture) SDL_DestroyTexture(texture);
                            texture = SDL_CreateTexture(sdlRenderer, SDL_PIXELFORMAT_RGB888,
                                SDL_TEXTUREACCESS_STREAMING, g_settings.renderWidth, g_settings.renderHeight);
                            SDL_SetTextureScaleMode(texture, SDL_ScaleModeNearest);
                            std::cout << "Render: " << g_settings.renderWidth << "x" << g_settings.renderHeight << "\n";
                            needsReset = true;
                            break;
                        
                        case SDLK_q:
                            g_settings.adjustWindowSize(false);
                            SDL_SetWindowSize(window, g_settings.windowWidth, g_settings.windowHeight);
                            break;
                        
                        case SDLK_e:
                            g_settings.adjustWindowSize(true);
                            SDL_SetWindowSize(window, g_settings.windowWidth, g_settings.windowHeight);
                            break;
                        
                        case SDLK_r:
                            g_settings.worldSeed = std::random_device{}();
                            world.generate(g_settings.worldSeed);
                            needsReset = true;
                            break;
                        
                        case SDLK_f:
                            g_settings.worldSeed++;
                            world.generate(g_settings.worldSeed);
                            needsReset = true;
                            break;
                        
                        case SDLK_t:
                            g_settings.timeOfDay -= 0.05f;
                            if (g_settings.timeOfDay < 0) g_settings.timeOfDay += 1.0f;
                            needsReset = true;
                            
                            std::cout << "Time Of Day: " << g_settings.timeOfDay << "\n";
                            break;
                        
                        case SDLK_g:
                            g_settings.timeOfDay += 0.05f;
                            if (g_settings.timeOfDay > 1) g_settings.timeOfDay -= 1.0f;
                            needsReset = true;
                            break;
                        
                        case SDLK_KP_1:
                            g_settings.enableCaustics = !g_settings.enableCaustics;
                            std::cout << "Caustics: " << (g_settings.enableCaustics ? "ON" : "OFF") << "\n";
                            needsReset = true;
                            break;
                        
                        case SDLK_KP_2:
                            g_settings.enableVolumetrics = !g_settings.enableVolumetrics;
                            std::cout << "Volumetrics: " << (g_settings.enableVolumetrics ? "ON" : "OFF") << "\n";
                            needsReset = true;
                            break;
                        
                        case SDLK_KP_3:
                            g_settings.causticQuality = (g_settings.causticQuality % 3) + 1;
                            std::cout << "Caustic Quality: ";
                            switch(g_settings.causticQuality) {
                                case 1: std::cout << "Low (8 samples)\n"; break;
                                case 2: std::cout << "Medium (16 samples)\n"; break;
                                case 3: std::cout << "High (32 samples)\n"; break;
                            }
                            needsReset = true;
                            break;
                    }
                } else if (event.type == SDL_MOUSEMOTION && 
                          (g_settings.mode == Settings::MODE_INTERACTIVE || 
                           g_settings.mode == Settings::MODE_RECORDING)) {
                    camera.yaw -= event.motion.xrel * 0.005f;
                    camera.pitch -= event.motion.yrel * 0.005f;
                    camera.pitch = std::max(-1.5f, std::min(1.5f, camera.pitch));
                    cameraMoving = true;
                }
            }
            
            // Movement (in interactive and recording modes)
            if (g_settings.mode == Settings::MODE_INTERACTIVE || 
                g_settings.mode == Settings::MODE_RECORDING) {
                float speed = 0.5f;
                Vec3 forward = camera.getForward();
                Vec3 right = camera.getRight();
                Vec3 oldPos = camera.position;
                
                if (keystate[SDL_SCANCODE_W]) camera.position = camera.position + forward * speed;
                if (keystate[SDL_SCANCODE_S]) camera.position = camera.position - forward * speed;
                if (keystate[SDL_SCANCODE_A]) camera.position = camera.position - right * speed;
                if (keystate[SDL_SCANCODE_D]) camera.position = camera.position + right * speed;
                if (keystate[SDL_SCANCODE_SPACE]) camera.position.y += speed;
                if (keystate[SDL_SCANCODE_LSHIFT]) camera.position.y -= speed;
                
                if (std::abs(camera.position.x - oldPos.x) > 0.001f ||
                    std::abs(camera.position.y - oldPos.y) > 0.001f ||
                    std::abs(camera.position.z - oldPos.z) > 0.001f ||
                    std::abs(camera.yaw - prevCamera.yaw) > 0.001f ||
                    std::abs(camera.pitch - prevCamera.pitch) > 0.001f) {
                    cameraMoving = true;
                }
            }
        }
        
        // Demo recording (time-based, not frame-based)
        if (g_settings.mode == Settings::MODE_RECORDING) {
            float recordInterval = 0.033f; // 30 Hz recording rate
            if (std::chrono::duration<float>(currentTime - lastRecordTime).count() >= recordInterval) {
                demoPath.addKeyframe(totalElapsed, camera.position.x, camera.position.y, 
                                    camera.position.z, camera.yaw, camera.pitch);
                lastRecordTime = currentTime;
            }
        }
        
        // Demo playback
        if (g_settings.mode == Settings::MODE_PLAYBACK || 
            g_settings.mode == Settings::MODE_BENCHMARK ||
            g_settings.mode == Settings::MODE_OFFLINE_RENDER) {
            
            if (g_settings.mode == Settings::MODE_OFFLINE_RENDER) {
                // Fixed time step for offline rendering
                demoTime = (offlineFrameCount / 30.0f); // 30 FPS output
            } else {
                demoTime += deltaTime;
            }
            
            float x, y, z, yaw, pitch;
            if (demoPath.getInterpolatedCamera(demoTime, x, y, z, yaw, pitch)) {
                camera.setFromKeyframe(x, y, z, yaw, pitch);
                
                // Only mark camera as moving for non-offline modes
                if (g_settings.mode != Settings::MODE_OFFLINE_RENDER) {
                    cameraMoving = true;
                }
            }
            
            // End conditions
            if (demoTime > demoPath.totalDuration) {
                if (g_settings.mode == Settings::MODE_BENCHMARK) {
                    benchmarkRecorder.totalTime = demoTime;
                    benchmarkRecorder.saveResults("benchmark_results.json");
                    std::cout << "Benchmark complete. Results saved to benchmark_results.json\n";
                    g_settings.mode = Settings::MODE_INTERACTIVE;
                } else if (g_settings.mode == Settings::MODE_OFFLINE_RENDER) {
                    std::cout << "Offline render complete. " << offlineFrameCount << " frames saved.\n";
                    running = false;
                } else {
                    demoTime = 0; // Loop demo
                }
            }
        }
        
        if (needsReset) {
            renderer.reset();
        }
        
        prevCamera = camera;
        
        // Render
        auto renderStart = std::chrono::high_resolution_clock::now();
        renderer.render(camera, world, cameraMoving || needsReset);
        auto renderEnd = std::chrono::high_resolution_clock::now();
        float renderTime = std::chrono::duration<float, std::milli>(renderEnd - renderStart).count();
        
        // Save frame for offline rendering
        if (g_settings.mode == Settings::MODE_OFFLINE_RENDER) {
            if (renderer.getSampleCount() >= g_settings.offlineTargetSamples) {
                std::stringstream ss;
                ss << g_settings.outputDir << "/frame_" << std::setfill('0') 
                   << std::setw(5) << offlineFrameCount << ".ppm";
                renderer.saveFrame(ss.str());
                std::cout << "Saved frame " << offlineFrameCount << " (samples: " 
                         << renderer.getSampleCount() << ")\n";
                offlineFrameCount++;
                renderer.reset();
            }
        }
        
        // Update display (skip for offline mode)
        if (!offlineMode) {
            SDL_UpdateTexture(texture, nullptr, renderer.getFramebuffer(), 
                            g_settings.renderWidth * sizeof(uint32_t));
            SDL_RenderClear(sdlRenderer);
            SDL_RenderCopy(sdlRenderer, texture, nullptr, nullptr);
            SDL_RenderPresent(sdlRenderer);
        }
        
        // FPS counter and benchmark recording
        frameCount++;
            
        if (std::chrono::duration<float>(currentTime - lastFPSTime).count() >= 1.0f) {
            currentFPS = frameCount;
            
            if (g_settings.mode == Settings::MODE_BENCHMARK) {
                benchmarkRecorder.recordFrame(demoTime, currentFPS, 
                                            renderer.getSampleCount(), renderTime);
            }
            
            if (!offlineMode) {
                std::cout << "FPS: " << frameCount << " (Samples: " 
                         << renderer.getSampleCount() << ")";
                
                switch (g_settings.mode) {
                    case Settings::MODE_RECORDING:
                        std::cout << " [RECORDING: " << demoPath.keyframes.size() << " keyframes]";
                        break;
                    case Settings::MODE_PLAYBACK:
                        std::cout << " [PLAYBACK: " << std::fixed << std::setprecision(1) 
                                 << (demoTime / demoPath.totalDuration * 100) << "%]";
                        break;
                    case Settings::MODE_BENCHMARK:
                        std::cout << " [BENCHMARK: " << std::fixed << std::setprecision(1) 
                                 << (demoTime / demoPath.totalDuration * 100) << "%]";
                        break;
                    default:
                        break;
                }
                
                if (g_settings.enableCaustics) {
                    std::cout << " [Caustics: ";
                    switch(g_settings.causticQuality) {
                        case 1: std::cout << "Low]"; break;
                        case 2: std::cout << "Med]"; break;
                        case 3: std::cout << "High]"; break;
                    }
                }
                
                std::cout << std::endl;
            }
            
            frameCount = 0;
            lastFPSTime = currentTime;
        }
    }
    
    // Cleanup
    if (texture) SDL_DestroyTexture(texture);
    if (sdlRenderer) SDL_DestroyRenderer(sdlRenderer);
    if (window) SDL_DestroyWindow(window);
    SDL_Quit();
    
    return 0;
}
