// CPU Pathtracer with Demo Recording & Benchmarking
// With Physically-Based Caustics

// MSVC: M_PI from <cmath>, and no min/max macros from <windows.h>
#define _USE_MATH_DEFINES
#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <atomic>
#include <chrono>
#include <algorithm>
#include <cstdio>
#include <sstream>
#include <iomanip>
#include <cstring>
#include <queue>
#include <set>
#include <string>
#include <tuple>
#include <immintrin.h>
#include <SDL2/SDL.h>
#include <fstream>
#include <filesystem>
#include <memory>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

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
        
        // Linear between keyframes: they are recorded at 30 Hz, and easing each
        // short segment would stop the camera at every keyframe (visible stutter).
        float span = kf2.time - kf1.time;
        float t = span > 1e-6f ? (time - kf1.time) / span : 0.0f;
        t = std::max(0.0f, std::min(1.0f, t));
        
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
        
        // Older recordings stamped the first keyframe with a stale clock value
        // (larger than the ones after it); it belongs at the start.
        if (keyframes.size() >= 2 && keyframes[0].time > keyframes[1].time) {
            keyframes[0].time = 0.0f;
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
    bool sampleLamps = true;          // sample light blocks directly (off: found by bounces only)
    bool enableParticles = true;      // drifting specks in the water
    float causticStrength = 1.0f;     // contrast of the caustic pattern (0 = even light)
    float shaftStrength = 1.0f;       // brightness of underwater light shafts
    int causticQuality = 2;  // caustic map detail: 1=low (2 texels per block), 2=medium (4), 3=high (8)
    
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
    int threads = 0;                  // 0 = one per hardware thread
    
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
constexpr float WATER_ANIM_SPEED = 1.5f;   // water animation units per second
constexpr int WATER_LEVEL = 11;            // water fills the blocks below this height; its surface is at y = WATER_LEVEL
constexpr float WATER_IOR = 1.333f;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================
// AVX2 SIMD kernels - 8-wide float math for the hot paths.
// Standard immintrin intrinsics only (no FMA), so a plain
// -mavx2 / /arch:AVX2 build works on both GCC and MSVC.
// ============================================================
struct F8 {
    __m256 v;
    F8() : v(_mm256_setzero_ps()) {}
    F8(__m256 x) : v(x) {}
    explicit F8(float x) : v(_mm256_set1_ps(x)) {}
    static F8 load(const float* p) { return F8(_mm256_loadu_ps(p)); }
    void store(float* p) const { _mm256_storeu_ps(p, v); }
    F8 operator+(F8 b) const { return F8(_mm256_add_ps(v, b.v)); }
    F8 operator-(F8 b) const { return F8(_mm256_sub_ps(v, b.v)); }
    F8 operator*(F8 b) const { return F8(_mm256_mul_ps(v, b.v)); }
    F8 operator/(F8 b) const { return F8(_mm256_div_ps(v, b.v)); }
    F8 operator-() const { return F8(_mm256_sub_ps(_mm256_setzero_ps(), v)); }
};
inline F8 min8(F8 a, F8 b) { return F8(_mm256_min_ps(a.v, b.v)); }
inline F8 max8(F8 a, F8 b) { return F8(_mm256_max_ps(a.v, b.v)); }
inline F8 sqrt8(F8 a) { return F8(_mm256_sqrt_ps(a.v)); }
inline F8 abs8(F8 a) { return F8(_mm256_andnot_ps(_mm256_set1_ps(-0.0f), a.v)); }
inline F8 floor8(F8 a) { return F8(_mm256_floor_ps(a.v)); }
inline F8 and8(F8 a, F8 b) { return F8(_mm256_and_ps(a.v, b.v)); }
inline F8 cmpge8(F8 a, F8 b) { return F8(_mm256_cmp_ps(a.v, b.v, _CMP_GE_OQ)); }
inline F8 cmple8(F8 a, F8 b) { return F8(_mm256_cmp_ps(a.v, b.v, _CMP_LE_OQ)); }
inline F8 cmplt8(F8 a, F8 b) { return F8(_mm256_cmp_ps(a.v, b.v, _CMP_LT_OQ)); }

// Horizontal sum of all 8 lanes
inline float hsum8(F8 a) {
    __m128 lo = _mm256_castps256_ps128(a.v);
    __m128 hi = _mm256_extractf128_ps(a.v, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 1));
    return _mm_cvtss_f32(lo);
}

// 8-wide simultaneous sin+cos, Cephes-style quadrant reduction
// (same accuracy class as scalar sinf/cosf for the argument ranges used here)
inline void sincos8(F8 x, F8& outSin, F8& outCos) {
    const __m256 signMask = _mm256_set1_ps(-0.0f);
    __m256 sinSign = _mm256_and_ps(x.v, signMask);
    __m256 ax = _mm256_andnot_ps(signMask, x.v);

    __m256 y = _mm256_mul_ps(ax, _mm256_set1_ps(1.27323954473516f)); // 4/pi
    __m256i j = _mm256_cvttps_epi32(y);
    j = _mm256_add_epi32(j, _mm256_set1_epi32(1));
    j = _mm256_and_si256(j, _mm256_set1_epi32(~1));
    y = _mm256_cvtepi32_ps(j);

    // sin sign flips when j&4; cos sign flips when (~(j-2))&4; polynomials swap when j&2
    __m256 swapSign = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_and_si256(j, _mm256_set1_epi32(4)), 29));
    __m256i jc = _mm256_andnot_si256(_mm256_sub_epi32(j, _mm256_set1_epi32(2)), _mm256_set1_epi32(4));
    __m256 cosSign = _mm256_castsi256_ps(_mm256_slli_epi32(jc, 29));
    sinSign = _mm256_xor_ps(sinSign, swapSign);
    __m256 polySwap = _mm256_castsi256_ps(_mm256_cmpeq_epi32(_mm256_and_si256(j, _mm256_set1_epi32(2)), _mm256_set1_epi32(2)));

    // Extended-precision argument reduction: ax = ((ax - y*DP1) - y*DP2) - y*DP3
    ax = _mm256_add_ps(ax, _mm256_mul_ps(y, _mm256_set1_ps(-0.78515625f)));
    ax = _mm256_add_ps(ax, _mm256_mul_ps(y, _mm256_set1_ps(-2.4187564849853515625e-4f)));
    ax = _mm256_add_ps(ax, _mm256_mul_ps(y, _mm256_set1_ps(-3.77489497744594108e-8f)));
    __m256 z = _mm256_mul_ps(ax, ax);

    __m256 sp = _mm256_set1_ps(-1.9515295891e-4f);
    sp = _mm256_add_ps(_mm256_mul_ps(sp, z), _mm256_set1_ps(8.3321608736e-3f));
    sp = _mm256_add_ps(_mm256_mul_ps(sp, z), _mm256_set1_ps(-1.6666654611e-1f));
    sp = _mm256_add_ps(_mm256_mul_ps(_mm256_mul_ps(sp, z), ax), ax);

    __m256 cp = _mm256_set1_ps(2.443315711809948e-5f);
    cp = _mm256_add_ps(_mm256_mul_ps(cp, z), _mm256_set1_ps(-1.388731625493765e-3f));
    cp = _mm256_add_ps(_mm256_mul_ps(cp, z), _mm256_set1_ps(4.166664568298827e-2f));
    cp = _mm256_mul_ps(cp, _mm256_mul_ps(z, z));
    cp = _mm256_add_ps(cp, _mm256_add_ps(_mm256_mul_ps(z, _mm256_set1_ps(-0.5f)), _mm256_set1_ps(1.0f)));

    __m256 sinPoly = _mm256_blendv_ps(sp, cp, polySwap);
    __m256 cosPoly = _mm256_blendv_ps(cp, sp, polySwap);
    outSin = F8(_mm256_xor_ps(sinPoly, sinSign));
    outCos = F8(_mm256_xor_ps(cosPoly, cosSign));
}

inline F8 sin8(F8 x) { F8 s, c; sincos8(x, s, c); return s; }

// 8-wide expf, Cephes-style
inline F8 exp8(F8 x) {
    __m256 xv = _mm256_min_ps(x.v, _mm256_set1_ps(88.3762626647949f));
    xv = _mm256_max_ps(xv, _mm256_set1_ps(-88.3762626647949f));
    __m256 fx = _mm256_floor_ps(_mm256_add_ps(_mm256_mul_ps(xv, _mm256_set1_ps(1.44269504088896341f)), _mm256_set1_ps(0.5f)));
    xv = _mm256_sub_ps(xv, _mm256_mul_ps(fx, _mm256_set1_ps(0.693359375f)));
    xv = _mm256_sub_ps(xv, _mm256_mul_ps(fx, _mm256_set1_ps(-2.12194440e-4f)));
    __m256 z = _mm256_mul_ps(xv, xv);
    __m256 y = _mm256_set1_ps(1.9875691500e-4f);
    y = _mm256_add_ps(_mm256_mul_ps(y, xv), _mm256_set1_ps(1.3981999507e-3f));
    y = _mm256_add_ps(_mm256_mul_ps(y, xv), _mm256_set1_ps(8.3334519073e-3f));
    y = _mm256_add_ps(_mm256_mul_ps(y, xv), _mm256_set1_ps(4.1665795894e-2f));
    y = _mm256_add_ps(_mm256_mul_ps(y, xv), _mm256_set1_ps(1.6666665459e-1f));
    y = _mm256_add_ps(_mm256_mul_ps(y, xv), _mm256_set1_ps(5.0000001201e-1f));
    y = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(y, z), xv), _mm256_set1_ps(1.0f));
    __m256i n = _mm256_cvttps_epi32(fx);
    n = _mm256_slli_epi32(_mm256_add_epi32(n, _mm256_set1_epi32(127)), 23);
    return F8(_mm256_mul_ps(y, _mm256_castsi256_ps(n)));
}

// Math utilities - Vec3 backed by a 128-bit SSE register (x, y, z, 0)
struct Vec3 {
    union {
        __m128 m;
        struct { float x, y, z, wPad; };
    };
    Vec3() : m(_mm_setzero_ps()) {}
    Vec3(float x, float y, float z) : m(_mm_set_ps(0.0f, z, y, x)) {}
    explicit Vec3(__m128 v) : m(v) {}

    Vec3 operator+(const Vec3& v) const { return Vec3(_mm_add_ps(m, v.m)); }
    Vec3 operator-(const Vec3& v) const { return Vec3(_mm_sub_ps(m, v.m)); }
    Vec3 operator*(float t) const { return Vec3(_mm_mul_ps(m, _mm_set1_ps(t))); }
    Vec3 operator*(const Vec3& v) const { return Vec3(_mm_mul_ps(m, v.m)); }
    Vec3 operator/(float t) const { return Vec3(_mm_div_ps(m, _mm_set1_ps(t))); }
    Vec3 operator-() const { return Vec3(_mm_sub_ps(_mm_setzero_ps(), m)); }

    float dot(const Vec3& v) const { return _mm_cvtss_f32(_mm_dp_ps(m, v.m, 0x71)); }
    Vec3 cross(const Vec3& v) const {
        __m128 aYzx = _mm_shuffle_ps(m, m, _MM_SHUFFLE(3, 0, 2, 1));
        __m128 bYzx = _mm_shuffle_ps(v.m, v.m, _MM_SHUFFLE(3, 0, 2, 1));
        __m128 c = _mm_sub_ps(_mm_mul_ps(m, bYzx), _mm_mul_ps(aYzx, v.m));
        return Vec3(_mm_shuffle_ps(c, c, _MM_SHUFFLE(3, 0, 2, 1)));
    }

    float length() const { return std::sqrt(dot(*this)); }
    Vec3 normalize() const {
        float l = length();
        return l > 0 ? *this / l : Vec3();
    }

    Vec3& operator+=(const Vec3& v) {
        m = _mm_add_ps(m, v.m);
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
    SAND,
    CORAL_PINK,
    CORAL_ORANGE,
    CORAL_PURPLE,
    KELP,
    SEA_LANTERN
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
    {{0.76f, 0.7f, 0.5f}, {0, 0, 0}, 0.9f, 1.0f, 0.0f, false}, // SAND
    {{0.90f, 0.36f, 0.48f}, {0, 0, 0}, 0.9f, 1.0f, 0.0f, false}, // CORAL_PINK
    {{0.95f, 0.52f, 0.18f}, {0, 0, 0}, 0.9f, 1.0f, 0.0f, false}, // CORAL_ORANGE
    {{0.58f, 0.32f, 0.80f}, {0, 0, 0}, 0.9f, 1.0f, 0.0f, false}, // CORAL_PURPLE
    {{0.20f, 0.46f, 0.16f}, {0, 0, 0}, 0.9f, 1.0f, 0.0f, false}, // KELP
    {{1, 1, 1}, {2.5f, 8.0f, 9.0f}, 0.1f, 1.0f, 0.0f, false}     // SEA_LANTERN
};

// Does this block give off light?
inline bool isEmitter(BlockType block) {
    const Vec3& e = g_materials[block].emission;
    return e.x > 0 || e.y > 0 || e.z > 0;
}

// Sun light system
struct SunLight {
    Vec3 direction;
    Vec3 color;
    float intensity;
    Vec3 refracted;       // direction of sunlight under flat water (unit, pointing down)
    float beamGain;       // strength of that beam relative to the sun in air
    
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

        // Under flat water: Snell refraction, Fresnel transmission. A horizontal
        // surface just under the water receives cosI * transmission of the
        // sun's light; beamGain turns that into the strength of the beam itself.
        float cosI = std::max(0.0f, -direction.y);
        float ratio = 1.0f / WATER_IOR;
        float cosT = std::sqrt(std::max(0.0f, 1.0f - ratio * ratio * (1.0f - cosI * cosI)));
        refracted = (direction * ratio + Vec3(0, 1, 0) * (ratio * cosI - cosT)).normalize();
        float fresnel = 0.02f + 0.98f * std::pow(1.0f - cosI, 5.0f);
        beamGain = cosI * (1.0f - fresnel) / std::max(0.05f, -refracted.y);
    }
    
    Vec3 getLightContribution() const {
        return color * intensity;
    }
};

// The sun for the frame being rendered (set once per pass by the renderer)
SunLight g_sun;

// Worker threads that stay alive between jobs. Every render pass, image
// conversion and caustic map build runs on them; starting two dozen new
// threads for each of those costs more than a small pass does.
class WorkerPool {
    std::vector<std::thread> workers;
    std::mutex mutex;
    std::condition_variable wake, finished;
    const std::function<void(int)>* job = nullptr;
    int jobThreads = 0;       // workers 0 .. jobThreads-1 take part in the current job
    int pending = 0;
    uint64_t jobId = 0;
    bool quit = false;

    void workerLoop(int index, uint64_t seenJob) {
        std::unique_lock<std::mutex> lock(mutex);
        while (true) {
            wake.wait(lock, [&] { return quit || jobId != seenJob; });
            if (quit) return;
            seenJob = jobId;
            if (index >= jobThreads) continue;
            const std::function<void(int)>* fn = job;
            lock.unlock();
            (*fn)(index);
            lock.lock();
            if (--pending == 0) finished.notify_one();
        }
    }

public:
    // Runs fn(0), fn(1), ... fn(threads - 1), each on its own thread, and waits for all of them
    void run(int threads, const std::function<void(int)>& fn) {
        threads = std::max(1, threads);
        std::unique_lock<std::mutex> lock(mutex);
        while (static_cast<int>(workers.size()) < threads) {
            int index = static_cast<int>(workers.size());
            workers.emplace_back([this, index, seen = jobId] { workerLoop(index, seen); });
        }
        job = &fn;
        jobThreads = threads;
        pending = threads;
        jobId++;
        wake.notify_all();
        finished.wait(lock, [&] { return pending == 0; });
        job = nullptr;
    }

    ~WorkerPool() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            quit = true;
        }
        wake.notify_all();
        for (auto& w : workers) w.join();
    }
};

WorkerPool g_pool;

inline int renderThreadCount() {
    return g_settings.threads > 0 ? g_settings.threads : int(std::max(1u, std::thread::hardware_concurrency()));
}

// Random utilities
// PCG32 (pcg-random.org, minimal variant). Each render thread re-seeds it for
// every pixel of every pass from (frame, pass, pixel), so an image depends only
// on its settings, never on how the threads happened to interleave.
struct Pcg32 {
    uint64_t state = 0x853c49e6748fea9bULL;
    uint64_t inc = 0xda3e39cb94b95bdbULL;

    void seed(uint64_t initState, uint64_t sequence) {
        state = 0;
        inc = (sequence << 1) | 1;
        next();
        state += initState;
        next();
    }
    uint32_t next() {
        uint64_t old = state;
        state = old * 6364136223846793005ULL + inc;
        uint32_t xorshifted = static_cast<uint32_t>(((old >> 18) ^ old) >> 27);
        uint32_t rot = static_cast<uint32_t>(old >> 59);
        return (xorshifted >> rot) | (xorshifted << ((32 - rot) & 31));
    }
};
thread_local Pcg32 rng;

// Uniform in [0, 1)
inline float random01() { return (rng.next() >> 8) * (1.0f / 16777216.0f); }

// Rays traced by the current thread (grid marches, shadow tests, packet lanes)
thread_local uint64_t t_rayCount = 0;

// Integer hashes for things that must be the same on every run and platform
inline uint32_t hash32(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352dU;
    x ^= x >> 15; x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}
inline uint32_t hashCell(int x, int y, int z, uint32_t salt) {
    return hash32(uint32_t(x) * 0x8da6b343U ^ uint32_t(y) * 0xd8163841U ^ uint32_t(z) * 0xcb1ab31fU ^ salt);
}
inline float hashFloat(uint32_t h) { return (h >> 8) * (1.0f / 16777216.0f); }

// Uniform direction on the unit sphere
inline Vec3 randomUnitVector() {
    float z = random01() * 2.0f - 1.0f;
    float phi = random01() * 2.0f * float(M_PI);
    float r = std::sqrt(std::max(0.0f, 1.0f - z * z));
    return Vec3(r * std::cos(phi), r * std::sin(phi), z);
}

// Cosine-weighted direction around a surface normal (the diffuse bounce:
// probability density cos(theta) / pi)
inline Vec3 randomCosineDirection(const Vec3& normal) {
    Vec3 dir = normal + randomUnitVector();
    return dir.dot(dir) > 1e-8f ? dir.normalize() : normal;
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

// Simple 3D value noise - all 8 cube-corner hashes computed in one 8-wide sin.
// Corner k adds its (dx,dy,dz) offsets pre-multiplied by the hash coefficients,
// so lane k holds hash(ix+dx, iy+dy, iz+dz) in n000..n111 order.
inline float noise3D(float x, float y, float z) {
    float ix = std::floor(x);
    float iy = std::floor(y);
    float iz = std::floor(z);

    float fx = x - ix;
    float fy = y - iy;
    float fz = z - iz;

    const __m256 cornerOfs = _mm256_setr_ps(
        0.0f,                            // (0,0,0)
        12.9898f,                        // (1,0,0)
        78.233f,                         // (0,1,0)
        12.9898f + 78.233f,              // (1,1,0)
        37.719f,                         // (0,0,1)
        12.9898f + 37.719f,              // (1,0,1)
        78.233f + 37.719f,               // (0,1,1)
        12.9898f + 78.233f + 37.719f);   // (1,1,1)
    float base = ix * 12.9898f + iy * 78.233f + iz * 37.719f;
    F8 n = sin8(F8(base) + F8(cornerOfs)) * F8(43758.5453f);
    F8 corners = n - floor8(n);
    alignas(32) float h[8];
    corners.store(h);

    // Smooth the fractional parts
    float sx = smoothstep(fx);
    float sy = smoothstep(fy);
    float sz = smoothstep(fz);

    // Trilinear interpolation
    float nx00 = lerp(h[0], h[1], sx);
    float nx10 = lerp(h[2], h[3], sx);
    float nx01 = lerp(h[4], h[5], sx);
    float nx11 = lerp(h[6], h[7], sx);

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

// Procedural coral texture: bumpy, with darker pores
inline Vec3 getCoralTexture(const Vec3& pos, const Vec3& base) {
    float bumps = noise3D(pos.x * 5.0f, pos.y * 5.0f, pos.z * 5.0f);
    float pores = noise3D(pos.x * 13.0f + 7.0f, pos.y * 13.0f, pos.z * 13.0f);
    Vec3 color = base * (0.7f + 0.5f * bumps);
    if (pores > 0.72f) color = color * 0.55f;
    return color;
}

// Procedural kelp texture: vertical blades with lighter and darker fronds
inline Vec3 getKelpTexture(const Vec3& pos, const Vec3& base) {
    float sway = std::sin(pos.y * 3.0f) * 1.5f;
    float blades = 0.5f + 0.5f * std::sin(pos.x * 14.0f + sway) * std::cos(pos.z * 14.0f + sway);
    float fronds = noise3D(pos.x * 4.0f, pos.y * 2.0f, pos.z * 4.0f);
    return base * (0.55f + 0.45f * blades + 0.3f * fronds);
}

// Voxel World
class World {
    std::vector<uint8_t> blocks;
    int seed;
    int topY = WORLD_HEIGHT - 1;      // highest y that holds any block
    std::vector<Vec3i> lights;        // every block that gives off light
    uint64_t generation = 0;          // changes whenever the world is regenerated

    // Sun horizon (see prepareSunShadows): a copy of the grid in which bit 7 of
    // a cell means "a ray toward the sun passing through here cannot hit any
    // block". Written by the main thread before the render threads start; they
    // only read it.
    static constexpr uint8_t SUN_CLEAR = 0x80;
    mutable std::vector<uint8_t> sunBlocks;
    mutable Vec3 sunClearDir;
    mutable uint64_t sunClearGeneration = 0;
    mutable bool sunClearValid = false;
    
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
    
    // Smooth 2D noise in [0, 1] from the integer hash: where kelp and coral grow in patches
    static float patchNoise(float x, float z, uint32_t salt) {
        int ix = static_cast<int>(std::floor(x)), iz = static_cast<int>(std::floor(z));
        float tx = x - ix, tz = z - iz;
        tx = tx * tx * (3.0f - 2.0f * tx);
        tz = tz * tz * (3.0f - 2.0f * tz);
        float a = hashFloat(hashCell(ix, 0, iz, salt)), b = hashFloat(hashCell(ix + 1, 0, iz, salt));
        float c = hashFloat(hashCell(ix, 0, iz + 1, salt)), d = hashFloat(hashCell(ix + 1, 0, iz + 1, salt));
        return (a * (1 - tx) + b * tx) * (1 - tz) + (c * (1 - tx) + d * tx) * tz;
    }

    // Rocks, coral, kelp and sea lanterns on the lake beds. Placement comes from
    // an integer hash of the column, so it is the same on every platform, and
    // it only ever replaces water: the land and the shorelines are unchanged.
    void decorateLakes() {
        const uint32_t salt = 0x9e3779b9U + uint32_t(seed) * 7919U;
        for (int z = 0; z < WORLD_SIZE; z++) {
            for (int x = 0; x < WORLD_SIZE; x++) {
                if (getBlock(x, WATER_LEVEL - 1, z) != WATER) continue;
                int bed = WATER_LEVEL - 1;                       // highest block of the bed
                while (bed >= 0 && getBlock(x, bed, z) == WATER) bed--;
                int depth = WATER_LEVEL - 1 - bed;               // blocks of water above the bed
                if (bed < 0 || depth < 2) continue;

                uint32_t h = hashCell(x, 0, z, salt);
                float r = hashFloat(h), r2 = hashFloat(hash32(h + 1)), r3 = hashFloat(hash32(h + 2));
                float kelpPatch = patchNoise(x * 0.11f, z * 0.11f, salt + 11);
                float coralPatch = patchNoise(x * 0.16f + 40.0f, z * 0.16f, salt + 23);

                if (depth >= 4 && r < 0.006f) {
                    setBlock(x, bed + 1, z, SEA_LANTERN);
                } else if (r < 0.035f) {
                    int height = std::min(depth - 1, 1 + (r2 > 0.6f ? 1 : 0));
                    for (int k = 0; k < height; k++) setBlock(x, bed + 1 + k, z, STONE);
                } else if (depth >= 3 && kelpPatch > 0.58f && r2 < 0.40f) {
                    int height = std::min(depth - 2, 1 + static_cast<int>(r3 * 5.0f));
                    for (int k = 0; k < height; k++) setBlock(x, bed + 1 + k, z, KELP);
                } else if (coralPatch > 0.60f && r2 < 0.45f) {
                    BlockType coral = r3 < 0.34f ? CORAL_PINK : r3 < 0.67f ? CORAL_ORANGE : CORAL_PURPLE;
                    int height = std::min(depth - 1, 1 + (hashFloat(hash32(h + 3)) > 0.7f ? 1 : 0));
                    for (int k = 0; k < height; k++) setBlock(x, bed + 1 + k, z, coral);
                }
            }
        }
    }

public:
    // +8 tail bytes so 4-byte SIMD gathers at the last block index stay in-bounds
    World() : blocks(WORLD_SIZE * WORLD_HEIGHT * WORLD_SIZE + 8, AIR), seed(42) {}
    
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
        
        generateWaterBodies(WATER_LEVEL);
        decorateLakes();

        topY = 0;
        for (int z = 0; z < WORLD_SIZE; z++)
            for (int y = 0; y < WORLD_HEIGHT; y++)
                for (int x = 0; x < WORLD_SIZE; x++)
                    if (getBlock(x, y, z) != AIR) topY = std::max(topY, y);
        buildLightCells();
        generation++;
    }

    uint64_t getGeneration() const { return generation; }

    // Shortcut for shadow rays toward the sun. Most of them pass through dozens
    // of empty cells to find nothing. For a sun direction that runs along one
    // horizontal axis (it always does here), sweep each row from its far end:
    // with `rise` the height a ray gains per column,
    // clear(x) = max(top(x), clear(next) - rise) is the highest any block further
    // along reaches when followed back down the ray's slope. A ray in a cell
    // above that, plus a margin, cannot hit anything, so those cells are flagged
    // in a copy of the grid and a march toward the sun stops at the first
    // flagged cell. The margin covers where the ray is inside its cell (up to
    // one column, so `rise`), a block's own height (1) and one more cell for
    // rounding in the march (1). The result is exactly what the full march
    // would have returned.
    void prepareSunShadows(const Vec3& toSun) const {
        if (sunClearValid && sunClearGeneration == generation && sunClearDir.x == toSun.x &&
            sunClearDir.y == toSun.y && sunClearDir.z == toSun.z) {
            return;
        }
        sunClearValid = false;
        sunClearDir = toSun;
        sunClearGeneration = generation;
        bool alongX = toSun.z == 0.0f && toSun.x != 0.0f;
        bool alongZ = toSun.x == 0.0f && toSun.z != 0.0f;
        bool vertical = toSun.x == 0.0f && toSun.z == 0.0f;
        if (!(toSun.y > 1e-3f) || !(alongX || alongZ || vertical)) return;    // no shortcut: rays march as usual

        sunBlocks = blocks;
        const float rise = vertical ? 0.0f : toSun.y / std::abs(alongX ? toSun.x : toSun.z);
        const float margin = rise + 2.001f;
        auto columnTop = [&](int x, int z) {        // highest solid block, -1 if none
            for (int y = topY; y >= 0; y--) {
                uint8_t b = blocks[x + y * WORLD_SIZE + z * WORLD_SIZE * WORLD_HEIGHT];
                if (b != AIR && b != WATER) return y;
            }
            return -1;
        };
        for (int line = 0; line < WORLD_SIZE; line++) {
            float clear = -1e30f;
            int step = alongX ? (toSun.x > 0 ? 1 : -1) : (toSun.z > 0 ? 1 : -1);
            // From the far end (in the sun's direction) back toward the near end
            for (int k = 0; k < WORLD_SIZE; k++) {
                int i = step > 0 ? WORLD_SIZE - 1 - k : k;
                int x = alongX ? i : line, z = alongX ? line : i;
                float top = float(columnTop(x, z));
                clear = vertical ? top : std::max(top, clear - rise);
                // Cells whose floor is above clear + margin
                float limit = clear + margin;
                int firstClear = limit < 0.0f ? 0 : static_cast<int>(std::floor(limit)) + 1;
                for (int y = firstClear; y < WORLD_HEIGHT; y++) {
                    sunBlocks[x + y * WORLD_SIZE + z * WORLD_SIZE * WORLD_HEIGHT] |= SUN_CLEAR;
                }
            }
        }
        sunClearValid = true;
    }

    // The grid to march through for a ray in direction `dir`: the flagged copy
    // toward the sun, the plain grid otherwise (no cell is flagged there)
    inline const uint8_t* gridFor(const Vec3& dir) const {
        bool sunward = sunClearValid && dir.x == sunClearDir.x && dir.y == sunClearDir.y && dir.z == sunClearDir.z;
        return sunward ? sunBlocks.data() : blocks.data();
    }

    // Is the water surface open above this column? (Outside the world counts
    // as open water, so light also arrives from beyond the edge.)
    bool waterSurfaceAt(int x, int z) const {
        if (x < 0 || x >= WORLD_SIZE || z < 0 || z >= WORLD_SIZE) return true;
        return blocks[x + (WATER_LEVEL - 1) * WORLD_SIZE + z * WORLD_SIZE * WORLD_HEIGHT] == WATER;
    }
    
    // Light blocks near a point, for sampling their light directly. Each
    // 8-block cell lists the light blocks closest to it (within LIGHT_RANGE).
    static constexpr int LIGHT_CELL = 8;
    static constexpr int MAX_CELL_LIGHTS = 8;
    static constexpr int CELLS_X = WORLD_SIZE / LIGHT_CELL;
    static constexpr int CELLS_Y = WORLD_HEIGHT / LIGHT_CELL;
    static constexpr int CELLS_Z = WORLD_SIZE / LIGHT_CELL;
    struct LightCell {
        int count = 0;
        int index[MAX_CELL_LIGHTS];
    };

    const LightCell& lightsNear(const Vec3& p) const {
        int cx = std::min(CELLS_X - 1, std::max(0, static_cast<int>(p.x) / LIGHT_CELL));
        int cy = std::min(CELLS_Y - 1, std::max(0, static_cast<int>(p.y) / LIGHT_CELL));
        int cz = std::min(CELLS_Z - 1, std::max(0, static_cast<int>(p.z) / LIGHT_CELL));
        return lightCells[cx + cy * CELLS_X + cz * CELLS_X * CELLS_Y];
    }
    const Vec3i& light(int i) const { return lights[i]; }

    std::vector<LightCell> lightCells;

    void buildLightCells() {
        const float LIGHT_RANGE = 28.0f;
        lights.clear();
        for (int z = 0; z < WORLD_SIZE; z++)
            for (int y = 0; y < WORLD_HEIGHT; y++)
                for (int x = 0; x < WORLD_SIZE; x++)
                    if (isEmitter(getBlock(x, y, z))) lights.emplace_back(x, y, z);

        lightCells.assign(CELLS_X * CELLS_Y * CELLS_Z, LightCell());
        std::vector<std::pair<float, int>> closest;
        for (int cz = 0; cz < CELLS_Z; cz++) {
            for (int cy = 0; cy < CELLS_Y; cy++) {
                for (int cx = 0; cx < CELLS_X; cx++) {
                    float px = (cx + 0.5f) * LIGHT_CELL, py = (cy + 0.5f) * LIGHT_CELL, pz = (cz + 0.5f) * LIGHT_CELL;
                    closest.clear();
                    for (int i = 0; i < static_cast<int>(lights.size()); i++) {
                        float dx = lights[i].x + 0.5f - px, dy = lights[i].y + 0.5f - py, dz = lights[i].z + 0.5f - pz;
                        float d = std::sqrt(dx * dx + dy * dy + dz * dz);
                        if (d < LIGHT_RANGE) closest.emplace_back(d, i);
                    }
                    std::sort(closest.begin(), closest.end());
                    LightCell& cell = lightCells[cx + cy * CELLS_X + cz * CELLS_X * CELLS_Y];
                    cell.count = std::min(MAX_CELL_LIGHTS, static_cast<int>(closest.size()));
                    for (int k = 0; k < cell.count; k++) cell.index[k] = closest[k].second;
                }
            }
        }
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
    
    // Clip a ray to the world box. Returns false if it never enters. For an
    // origin outside the world, `pos` moves to the entry point, `entryDist` is
    // the distance to it and `entryNormal` the face it enters through.
    static bool enterWorld(Vec3& pos, const Vec3& dir, float maxDist, float& entryDist, Vec3& entryNormal) {
        entryDist = 0.0f;
        const float lo[3] = {0.0f, 0.0f, 0.0f};
        const float hi[3] = {float(WORLD_SIZE), float(WORLD_HEIGHT), float(WORLD_SIZE)};
        const float o[3] = {pos.x, pos.y, pos.z};
        const float d[3] = {dir.x, dir.y, dir.z};
        if (o[0] >= lo[0] && o[0] < hi[0] && o[1] >= lo[1] && o[1] < hi[1] && o[2] >= lo[2] && o[2] < hi[2])
            return true;
        float tEnter = 0.0f, tExit = maxDist;
        int axis = -1;
        for (int a = 0; a < 3; a++) {
            if (d[a] == 0.0f) {
                if (o[a] < lo[a] || o[a] >= hi[a]) return false;
                continue;
            }
            float t0 = (lo[a] - o[a]) / d[a];
            float t1 = (hi[a] - o[a]) / d[a];
            if (t0 > t1) std::swap(t0, t1);
            if (t0 > tEnter) { tEnter = t0; axis = a; }
            tExit = std::min(tExit, t1);
        }
        if (axis < 0 || tEnter >= tExit) return false;
        entryDist = tEnter + 1e-3f;                 // just inside the face
        pos = pos + dir * entryDist;
        float n[3] = {0.0f, 0.0f, 0.0f};
        n[axis] = d[axis] > 0 ? -1.0f : 1.0f;
        entryNormal = Vec3(n[0], n[1], n[2]);
        return true;
    }

    bool raycast(const Ray& ray, float maxDist, Vec3& hitPos, Vec3& hitNormal, BlockType& hitBlock) const {
        Vec3 pos = ray.origin;
        Vec3 dir = ray.direction;
        hitBlock = AIR;                             // defined even when nothing is hit
        t_rayCount++;
        Vec3 normal(0, 1, 0);
        float entryDist;
        if (!enterWorld(pos, dir, maxDist, entryDist, normal)) return false;
        maxDist -= entryDist;
        
        int x = std::min(WORLD_SIZE - 1, std::max(0, static_cast<int>(std::floor(pos.x))));
        int y = std::min(WORLD_HEIGHT - 1, std::max(0, static_cast<int>(std::floor(pos.y))));
        int z = std::min(WORLD_SIZE - 1, std::max(0, static_cast<int>(std::floor(pos.z))));
        
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

        BlockType startBlock = getBlock(static_cast<int>(std::floor(ray.origin.x)),
                                        static_cast<int>(std::floor(ray.origin.y)),
                                        static_cast<int>(std::floor(ray.origin.z)));
        bool startedInWater = (startBlock == WATER);
        bool currentlyInWater = startedInWater;

        // The cell is always inside the world here (clamped at the start, and
        // the loop ends when a step leaves it), so the grid is read directly
        // and its index moves with the cell.
        const uint8_t* grid = blocks.data();
        int idx = x + y * WORLD_SIZE + z * WORLD_SIZE * WORLD_HEIGHT;
        const int idxStepX = stepX, idxStepY = stepY * WORLD_SIZE, idxStepZ = stepZ * WORLD_SIZE * WORLD_HEIGHT;
        int axis = -1;              // axis of the last step: the face the ray came in through (-1: none yet)
        auto hit = [&](BlockType block) {
            hitPos = pos + dir * dist;
            hitBlock = block;
            hitNormal = axis == 0 ? Vec3(-stepX, 0, 0) : axis == 1 ? Vec3(0, -stepY, 0)
                      : axis == 2 ? Vec3(0, 0, -stepZ) : normal;
            return true;
        };

        while (dist < maxDist) {
            BlockType block = static_cast<BlockType>(grid[idx]);

            // A water surface, crossed from either side
            if (currentlyInWater ? block == AIR : block == WATER) return hit(WATER);
            if (block != AIR && block != WATER) return hit(block);

            if (tMaxX < tMaxY) {
                if (tMaxX < tMaxZ) {
                    x += stepX;
                    idx += idxStepX;
                    dist = tMaxX;
                    tMaxX += tDeltaX;
                    axis = 0;
                } else {
                    z += stepZ;
                    idx += idxStepZ;
                    dist = tMaxZ;
                    tMaxZ += tDeltaZ;
                    axis = 2;
                }
            } else {
                if (tMaxY < tMaxZ) {
                    y += stepY;
                    idx += idxStepY;
                    dist = tMaxY;
                    tMaxY += tDeltaY;
                    axis = 1;
                } else {
                    z += stepZ;
                    idx += idxStepZ;
                    dist = tMaxZ;
                    tMaxZ += tDeltaZ;
                    axis = 2;
                }
            }

            if (x < 0 || x >= WORLD_SIZE || y < 0 || y >= WORLD_HEIGHT || z < 0 || z >= WORLD_SIZE) {
                break;
            }
            if (stepY > 0 && y > topY) break;       // rising above every block: open sky
        }

        return false;
    }

    // First solid block along a ray, or AIR if there is none within maxDist.
    // Water is transparent here: light passes through it, so blocks above a
    // lake shade its bed and a ray that reaches open sky is never read as blocked.
    BlockType firstSolid(const Vec3& origin, const Vec3& dir, float maxDist) const {
        t_rayCount++;
        Vec3 pos = origin;
        Vec3 normal;
        float entryDist;
        if (!enterWorld(pos, dir, maxDist, entryDist, normal)) return AIR;
        maxDist -= entryDist;

        int x = std::min(WORLD_SIZE - 1, std::max(0, static_cast<int>(std::floor(pos.x))));
        int y = std::min(WORLD_HEIGHT - 1, std::max(0, static_cast<int>(std::floor(pos.y))));
        int z = std::min(WORLD_SIZE - 1, std::max(0, static_cast<int>(std::floor(pos.z))));
        const uint8_t* grid = gridFor(dir);

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
        while (dist < maxDist) {
            uint8_t block = grid[x + y * WORLD_SIZE + z * WORLD_SIZE * WORLD_HEIGHT];
            if (block & SUN_CLEAR) return AIR;      // toward the sun, above everything in the way
            if (block != AIR && block != WATER) return static_cast<BlockType>(block);

            if (tMaxX < tMaxY && tMaxX < tMaxZ) {
                x += stepX; dist = tMaxX; tMaxX += tDeltaX;
            } else if (tMaxY < tMaxZ) {
                y += stepY; dist = tMaxY; tMaxY += tDeltaY;
            } else {
                z += stepZ; dist = tMaxZ; tMaxZ += tDeltaZ;
            }
            if (x < 0 || x >= WORLD_SIZE || y < 0 || y >= WORLD_HEIGHT || z < 0 || z >= WORLD_SIZE) break;
            if (stepY > 0 && y > topY) break;       // rising above every block: open sky
        }
        return AIR;
    }

    // Sun shadow test: true if a solid block lies between `origin` and the sun.
    bool sunOccluded(const Vec3& origin, const Vec3& dir, float maxDist) const {
        return firstSolid(origin, dir, maxDist) != AIR;
    }

    // First solid block a volumetric shadow ray hits (AIR if none): the
    // one-ray version of raycastShadow8, with the same rules.
    uint8_t shadowBlock(const Vec3& origin, const Vec3& dir, float maxDist) const {
        if (origin.x < 0 || origin.x >= WORLD_SIZE || origin.y < 0 || origin.y >= WORLD_HEIGHT ||
            origin.z < 0 || origin.z >= WORLD_SIZE) {
            return AIR;                             // the packet version does not enter from outside
        }
        return firstSolid(origin, dir, maxDist);
    }

    // 8-wide shadow raycast: 8 origins, one shared direction (volumetric shadow
    // rays all point at the sun, so the DDA steps/deltas are uniform and only
    // per-lane voxel coords and tMax values diverge). Lanes march in lockstep
    // with masked termination. Water is transparent (sunlight passes through
    // it): outBlock[i] is the first solid block, or AIR when nothing is hit
    // within maxDist / the ray leaves the world.
    void raycastShadow8(const float* ox, const float* oy, const float* oz,
                        const Vec3& dirIn, float maxDist,
                        const bool laneActive[8], uint8_t outBlock[8]) const {
        Vec3 dir = dirIn.normalize();
        const uint8_t* bp = gridFor(dirIn);
        const __m256i widthV = _mm256_set1_epi32(WORLD_SIZE);
        const __m256i heightV = _mm256_set1_epi32(WORLD_HEIGHT);
        const __m256i minusOne = _mm256_set1_epi32(-1);
        const __m256i maxIdx = _mm256_set1_epi32(WORLD_SIZE * WORLD_HEIGHT * WORLD_SIZE - 1);
        const __m256i waterV = _mm256_set1_epi32(WATER);
        const __m256i clearBit = _mm256_set1_epi32(SUN_CLEAR);

        F8 pox = F8::load(ox), poy = F8::load(oy), poz = F8::load(oz);
        F8 fx = floor8(pox), fy = floor8(poy), fz = floor8(poz);
        __m256i x = _mm256_cvttps_epi32(fx.v);
        __m256i y = _mm256_cvttps_epi32(fy.v);
        __m256i z = _mm256_cvttps_epi32(fz.v);

        int stepX = dir.x > 0 ? 1 : -1;
        int stepY = dir.y > 0 ? 1 : -1;
        int stepZ = dir.z > 0 ? 1 : -1;

        F8 tMaxX = (dir.x != 0) ? (fx + F8(stepX > 0 ? 1.0f : 0.0f) - pox) / F8(dir.x) : F8(1e30f);
        F8 tMaxY = (dir.y != 0) ? (fy + F8(stepY > 0 ? 1.0f : 0.0f) - poy) / F8(dir.y) : F8(1e30f);
        F8 tMaxZ = (dir.z != 0) ? (fz + F8(stepZ > 0 ? 1.0f : 0.0f) - poz) / F8(dir.z) : F8(1e30f);
        const __m256 tDeltaX = _mm256_set1_ps((dir.x != 0) ? stepX / dir.x : 1e30f);
        const __m256 tDeltaY = _mm256_set1_ps((dir.y != 0) ? stepY / dir.y : 1e30f);
        const __m256 tDeltaZ = _mm256_set1_ps((dir.z != 0) ? stepZ / dir.z : 1e30f);
        const __m256 maxDistV = _mm256_set1_ps(maxDist);
        const __m256i stepXV = _mm256_set1_epi32(stepX), stepYV = _mm256_set1_epi32(stepY), stepZV = _mm256_set1_epi32(stepZ);
        // The block index moves with the cell, so it is never recomputed from x, y, z
        const __m256i idxStepX = _mm256_set1_epi32(stepX);
        const __m256i idxStepY = _mm256_set1_epi32(stepY * WORLD_SIZE);
        const __m256i idxStepZ = _mm256_set1_epi32(stepZ * WORLD_SIZE * WORLD_HEIGHT);
        const __m256i topV = _mm256_set1_epi32(topY + 1);

        auto inBoundsMask = [&]() {
            __m256i okX = _mm256_and_si256(_mm256_cmpgt_epi32(x, minusOne), _mm256_cmpgt_epi32(widthV, x));
            __m256i okY = _mm256_and_si256(_mm256_cmpgt_epi32(y, minusOne), _mm256_cmpgt_epi32(heightV, y));
            __m256i okZ = _mm256_and_si256(_mm256_cmpgt_epi32(z, minusOne), _mm256_cmpgt_epi32(widthV, z));
            return _mm256_and_si256(okX, _mm256_and_si256(okY, okZ));
        };

        __m256i active = _mm256_setr_epi32(
            laneActive[0] ? -1 : 0, laneActive[1] ? -1 : 0, laneActive[2] ? -1 : 0, laneActive[3] ? -1 : 0,
            laneActive[4] ? -1 : 0, laneActive[5] ? -1 : 0, laneActive[6] ? -1 : 0, laneActive[7] ? -1 : 0);
        __m256i result = _mm256_setzero_si256();    // AIR = no hit
        for (int i = 0; i < 8; i++) t_rayCount += laneActive[i] ? 1 : 0;

        __m256i idx = _mm256_add_epi32(x, _mm256_add_epi32(_mm256_mullo_epi32(y, _mm256_set1_epi32(WORLD_SIZE)),
                                                           _mm256_mullo_epi32(z, _mm256_set1_epi32(WORLD_SIZE * WORLD_HEIGHT))));
        __m256i inBounds = inBoundsMask();          // for the current cells; carried from step to step

        for (int guard = 0; guard < 2048; guard++) {
            if (_mm256_movemask_ps(_mm256_castsi256_ps(active)) == 0) break;

            __m256i safeIdx = _mm256_max_epi32(_mm256_min_epi32(idx, maxIdx), _mm256_setzero_si256());
            __m256i raw = _mm256_i32gather_epi32((const int*)bp, safeIdx, 1);
            // Out-of-bounds reads as AIR, like getBlock()
            __m256i block = _mm256_and_si256(_mm256_and_si256(raw, _mm256_set1_epi32(0xFF)), inBounds);
            // Toward the sun and above everything in the way: the lane is done, nothing hit
            __m256i clear = _mm256_cmpeq_epi32(_mm256_and_si256(block, clearBit), clearBit);
            active = _mm256_andnot_si256(clear, active);

            __m256i isAir = _mm256_cmpeq_epi32(block, _mm256_setzero_si256());
            __m256i isWater = _mm256_cmpeq_epi32(block, waterV);
            __m256i isSolid = _mm256_andnot_si256(_mm256_or_si256(isAir, isWater), minusOne);
            __m256i hit = _mm256_and_si256(active, isSolid);
            result = _mm256_blendv_epi8(result, block, hit);
            active = _mm256_andnot_si256(hit, active);

            // DDA step: same tie-breaking as the scalar version
            __m256 ltXY = _mm256_cmp_ps(tMaxX.v, tMaxY.v, _CMP_LT_OQ);
            __m256 ltXZ = _mm256_cmp_ps(tMaxX.v, tMaxZ.v, _CMP_LT_OQ);
            __m256 ltYZ = _mm256_cmp_ps(tMaxY.v, tMaxZ.v, _CMP_LT_OQ);
            __m256 maskX = _mm256_and_ps(ltXY, ltXZ);
            __m256 maskY = _mm256_andnot_ps(ltXY, ltYZ);
            __m256 maskZ = _mm256_andnot_ps(_mm256_or_ps(maskX, maskY), _mm256_castsi256_ps(minusOne));
            __m256i mX = _mm256_castps_si256(maskX), mY = _mm256_castps_si256(maskY), mZ = _mm256_castps_si256(maskZ);

            __m256 dist = _mm256_blendv_ps(tMaxZ.v, tMaxY.v, maskY);
            dist = _mm256_blendv_ps(dist, tMaxX.v, maskX);

            x = _mm256_add_epi32(x, _mm256_and_si256(stepXV, mX));
            y = _mm256_add_epi32(y, _mm256_and_si256(stepYV, mY));
            z = _mm256_add_epi32(z, _mm256_and_si256(stepZV, mZ));
            idx = _mm256_add_epi32(idx, _mm256_or_si256(_mm256_and_si256(idxStepX, mX),
                                        _mm256_or_si256(_mm256_and_si256(idxStepY, mY), _mm256_and_si256(idxStepZ, mZ))));
            tMaxX = F8(_mm256_blendv_ps(tMaxX.v, _mm256_add_ps(tMaxX.v, tDeltaX), maskX));
            tMaxY = F8(_mm256_blendv_ps(tMaxY.v, _mm256_add_ps(tMaxY.v, tDeltaY), maskY));
            tMaxZ = F8(_mm256_blendv_ps(tMaxZ.v, _mm256_add_ps(tMaxZ.v, tDeltaZ), maskZ));

            active = _mm256_and_si256(active, _mm256_castps_si256(_mm256_cmp_ps(dist, maxDistV, _CMP_LT_OQ)));
            inBounds = inBoundsMask();
            active = _mm256_and_si256(active, inBounds);
            // Rising above every block: open sky, the lane is done
            if (stepY > 0) active = _mm256_and_si256(active, _mm256_cmpgt_epi32(topV, y));
        }

        alignas(32) int res[8];
        _mm256_store_si256((__m256i*)res, result);
        for (int i = 0; i < 8; i++) outBlock[i] = (uint8_t)res[i];
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

// The water surface is a sum of eight waves travelling in different
// directions, from 5-block swells to half-block ripples. Each is
// height = a * sin(kx * x + kz * z + speed * t); only its slope is needed:
// (slopeX, slopeZ) * cos(...). Long waves are gentle, so sunlight keeps
// focusing well below the surface. Directions that are not multiples of each
// other keep the caustics an irregular network, not a grid.
struct WaterWave {
    float kx, kz, speed, slopeX, slopeZ;
};
static const WaterWave g_waterWaves[8] = {
    {  1.1140f,   0.4055f,  1.307f,   0.0940f,   0.0342f},   //   20 deg, wavelength 5.3, slope 0.100
    {  1.3197f,  -0.9241f,  1.523f,   0.0901f,  -0.0631f},   //  -35 deg, wavelength 3.9, slope 0.110
    {  0.5246f,   1.9578f,  1.708f,   0.0259f,   0.0966f},   //   75 deg, wavelength 3.1, slope 0.100
    { -1.7560f,   2.0927f,  1.983f,  -0.0579f,   0.0689f},   //  130 deg, wavelength 2.3, slope 0.090
    {  0.6418f,  -3.6398f,  2.307f,   0.0139f,  -0.0788f},   //  -80 deg, wavelength 1.7, slope 0.080
    { -4.6685f,   1.2509f,  2.638f,  -0.0676f,   0.0181f},   //  165 deg, wavelength 1.3, slope 0.070
    {  4.4875f,   5.3480f,  3.171f,   0.0321f,   0.0383f},   //   50 deg, wavelength 0.9, slope 0.050
    { -5.2360f,  -9.0690f,  3.883f,  -0.0175f,  -0.0303f},   // -120 deg, wavelength 0.6, slope 0.035
};

// Water surface slope (dh/dx, dh/dz) at 8 points at once. The same function
// shades the surface and drives the caustic map, so the two always agree.
inline void waterSlope8(F8 px, F8 pz, float time, F8& dx, F8& dz) {
    dx = F8(0.0f);
    dz = F8(0.0f);
    for (const WaterWave& w : g_waterWaves) {
        F8 s, c;
        sincos8(px * F8(w.kx) + pz * F8(w.kz) + F8(time * w.speed), s, c);
        dx = dx + F8(w.slopeX) * c;
        dz = dz + F8(w.slopeZ) * c;
    }
}

// Water surface normals (pointing up) at 8 points
inline void waterNormal8(F8 px, F8 pz, float time, F8& nx, F8& ny, F8& nz) {
    F8 dx, dz;
    waterSlope8(px, pz, time, dx, dz);
    F8 invLen = F8(1.0f) / sqrt8(dx * dx + dz * dz + F8(1.0f));
    nx = -dx * invLen;
    ny = invLen;
    nz = -dz * invLen;
}

// Water surface normal (pointing up) at one point
inline Vec3 getWaterNormal(const Vec3& pos, float time) {
    F8 dx, dz;
    waterSlope8(F8(pos.x), F8(pos.z), time, dx, dz);
    return Vec3(-_mm256_cvtss_f32(dx.v), 1.0f, -_mm256_cvtss_f32(dz.v)).normalize();
}

// Get sky color
// withSun adds the sun's disc and glow: for rays the camera sees. Bounce rays
// leave it out, because surfaces already receive the sun as direct light.
Vec3 getSkyColor(const Vec3& direction, float timeOfDay, const SunLight& sun, bool withSun = true) {
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
    if (!withSun) return skyGradient;

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

// ============================================================
// Water as a medium
// ============================================================

// Light lost per block of water travelled, per color. Red goes first, which is
// what turns shallow water turquoise and deep water blue.
static const float WATER_EXTINCTION[3] = {0.150f, 0.060f, 0.038f};
// Light scattered many times inside the water: the color a long look through it fades to
static const Vec3 WATER_GLOW(0.035f, 0.300f, 0.480f);
constexpr float WATER_SCATTER = 0.300f;        // single scattering toward the eye (light shafts)
constexpr float PARTICLE_BRIGHTNESS = 2.5f;
// Artistic: sunlight under water is shown brighter than it physically is (as
// if the eye had adapted), so caustics and shafts read as bright, dancing light.
constexpr float UNDERWATER_SUN_GAIN = 3.0f;

// Fraction of light that survives `dist` blocks of water, per color
inline Vec3 waterTransmittance(float dist) {
    alignas(32) float e[8] = {-WATER_EXTINCTION[0] * dist, -WATER_EXTINCTION[1] * dist,
                              -WATER_EXTINCTION[2] * dist, 0, 0, 0, 0, 0};
    exp8(F8::load(e)).store(e);
    return Vec3(e[0], e[1], e[2]);
}

// The water's own glow at a height: brighter near the surface, tied to the sun
inline Vec3 waterGlow(float y, const SunLight& sun) {
    float depth = std::max(0.0f, float(WATER_LEVEL) - y);
    return WATER_GLOW * (sun.intensity * (0.35f + 0.65f * std::exp(-depth * 0.12f)));
}

// ============================================================
// Caustic map
//
// Sunlight is refracted by the waves and lands unevenly on whatever is below:
// bright lines where the surface focuses it, dimmer patches between. Once per
// frame, light is sent down from a fine grid of points on the surface and
// collected in a stack of horizontal layers, one per block of depth. A value
// of 1 means "as much light as under flat water"; focused lines are above 1.
// Lookups are then a few texture reads, with no noise, at any depth.
// ============================================================
class CausticMap {
public:
    static constexpr int MARGIN = 12;                 // blocks of surface simulated beyond the world edge
    static constexpr int LAYERS = WATER_LEVEL + 1;    // depths 0 .. WATER_LEVEL

    bool ready() const { return !data.empty(); }
    int texels() const { return size; }
    const float* layer(int d) const { return &data[size_t(d) * size * size]; }

    void update(const World& world, const SunLight& sun, float time, int texelsPerBlock) {
        if (ready() && res == texelsPerBlock && time == builtTime && world.getGeneration() == builtWorld &&
            sun.direction.x == builtSun.x && sun.direction.y == builtSun.y && sun.direction.z == builtSun.z) {
            return;
        }
        res = texelsPerBlock;
        size = WORLD_SIZE * res;
        builtTime = time;
        builtWorld = world.getGeneration();
        builtSun = sun.direction;
        data.assign(size_t(LAYERS) * size * size, 0.0f);

        // Photons: a grid over the surface, 2 x 2 per texel
        const int perTexel = 2;
        const float spacing = 1.0f / float(res * perTexel);
        const int np = (WORLD_SIZE + 2 * MARGIN) * res * perTexel;        // per side; a multiple of 8
        offX.resize(size_t(np) * np);
        offZ.resize(size_t(np) * np);
        weight.resize(size_t(np) * np);

        const Vec3 incident = sun.direction;
        const float ratio = 1.0f / WATER_IOR;
        const float cosFlat = std::max(1e-3f, -incident.y);
        const float flatFlux = cosFlat * (1.0f - (0.02f + 0.98f * std::pow(1.0f - cosFlat, 5.0f)));

        int threads = renderThreadCount();

        // 1. Refract every photon through the surface: where it lands per block
        //    of depth, and how much light it carries relative to flat water.
        auto refractRows = [&](int j0, int j1) {
            alignas(32) float wxs[8], mask[8], ox[8], oz[8], wg[8];
            for (int j = j0; j < j1; j++) {
                float wz = -float(MARGIN) + (j + 0.5f) * spacing;
                int bz = static_cast<int>(std::floor(wz));
                for (int i = 0; i < np; i += 8) {
                    bool any = false;
                    for (int k = 0; k < 8; k++) {
                        wxs[k] = -float(MARGIN) + (i + k + 0.5f) * spacing;
                        bool water = world.waterSurfaceAt(static_cast<int>(std::floor(wxs[k])), bz);
                        mask[k] = water ? 1.0f : 0.0f;
                        any = any || water;
                    }
                    float* dox = &offX[size_t(j) * np + i];
                    float* doz = &offZ[size_t(j) * np + i];
                    float* dw = &weight[size_t(j) * np + i];
                    if (!any) {
                        for (int k = 0; k < 8; k++) { dox[k] = 0; doz[k] = 0; dw[k] = 0; }
                        continue;
                    }
                    F8 nx, ny, nz;
                    waterNormal8(F8::load(wxs), F8(wz), time, nx, ny, nz);
                    F8 cosI = -(nx * F8(incident.x) + ny * F8(incident.y) + nz * F8(incident.z));
                    F8 valid = cmpge8(cosI, F8(0.01f));                       // grazing light is reflected away
                    F8 sinT2 = F8(ratio * ratio) * (F8(1.0f) - cosI * cosI);
                    F8 cosT = sqrt8(max8(F8(1.0f) - sinT2, F8(0.0f)));
                    F8 rc = F8(ratio) * cosI - cosT;
                    F8 rx = F8(incident.x * ratio) + nx * rc;
                    F8 ry = F8(incident.y * ratio) + ny * rc;
                    F8 rz = F8(incident.z * ratio) + nz * rc;
                    valid = and8(valid, cmplt8(ry, F8(-0.01f)));              // must head down
                    F8 invDown = F8(1.0f) / max8(-ry, F8(0.01f));
                    // Flux through a tilted patch of surface: cos(incidence) * area (1 / ny), times transmission
                    F8 om = F8(1.0f) - min8(cosI, F8(1.0f));
                    F8 om2 = om * om;
                    F8 fresnel = F8(0.02f) + F8(0.98f) * om2 * om2 * om;
                    F8 flux = cosI * (F8(1.0f) - fresnel) / max8(ny, F8(0.05f)) * F8(1.0f / flatFlux);
                    (rx * invDown).store(ox);
                    (rz * invDown).store(oz);
                    and8(flux, valid).store(wg);
                    for (int k = 0; k < 8; k++) { dox[k] = ox[k]; doz[k] = oz[k]; dw[k] = wg[k] * mask[k]; }
                }
            }
        };
        runParallel(np, threads, refractRows);

        // 2. One layer per depth: drop each photon where it lands, then soften.
        auto buildLayers = [&](int d0, int d1) {
            std::vector<float> tmp(size_t(size) * size);
            for (int d = d0; d < d1; d++) {
                float* layer = &data[size_t(d) * size * size];
                const float scale = 1.0f / float(perTexel * perTexel);
                for (int j = 0; j < np; j++) {
                    float wz = -float(MARGIN) + (j + 0.5f) * spacing;
                    const float* pox = &offX[size_t(j) * np];
                    const float* poz = &offZ[size_t(j) * np];
                    const float* pw = &weight[size_t(j) * np];
                    for (int i = 0; i < np; i++) {
                        float w = pw[i];
                        if (w <= 0.0f) continue;
                        float wx = -float(MARGIN) + (i + 0.5f) * spacing;
                        float fx = (wx + pox[i] * d) * res - 0.5f;
                        float fz = (wz + poz[i] * d) * res - 0.5f;
                        int ix = static_cast<int>(std::floor(fx));
                        int iz = static_cast<int>(std::floor(fz));
                        if (ix < -1 || ix >= size || iz < -1 || iz >= size) continue;
                        float tx = fx - ix, tz = fz - iz;
                        w *= scale;
                        if (iz >= 0) {
                            if (ix >= 0) layer[size_t(iz) * size + ix] += w * (1 - tx) * (1 - tz);
                            if (ix + 1 < size) layer[size_t(iz) * size + ix + 1] += w * tx * (1 - tz);
                        }
                        if (iz + 1 < size) {
                            if (ix >= 0) layer[size_t(iz + 1) * size + ix] += w * (1 - tx) * tz;
                            if (ix + 1 < size) layer[size_t(iz + 1) * size + ix + 1] += w * tx * tz;
                        }
                    }
                }
                // The sun is not a point and water scatters: the pattern softens with depth
                blur(layer, tmp.data(), res * (0.04f + 0.014f * d));
            }
        };
        runParallel(LAYERS, threads, buildLayers);
    }

    // Light at (x, z) and a depth below the surface, relative to flat water
    float sample(float x, float z, float depth) const {
        float fd = std::min(float(LAYERS - 1), std::max(0.0f, depth));
        int d0 = std::min(LAYERS - 2, static_cast<int>(fd));
        float td = fd - d0;
        float fx = std::min(float(size - 1), std::max(0.0f, x * res - 0.5f));
        float fz = std::min(float(size - 1), std::max(0.0f, z * res - 0.5f));
        int ix = std::min(size - 2, static_cast<int>(fx));
        int iz = std::min(size - 2, static_cast<int>(fz));
        float tx = fx - ix, tz = fz - iz;
        const float* a = &data[(size_t(d0) * size + iz) * size + ix];
        const float* b = a + size_t(size) * size;
        float la = (a[0] * (1 - tx) + a[1] * tx) * (1 - tz) + (a[size] * (1 - tx) + a[size + 1] * tx) * tz;
        float lb = (b[0] * (1 - tx) + b[1] * tx) * (1 - tz) + (b[size] * (1 - tx) + b[size + 1] * tx) * tz;
        return la * (1 - td) + lb * td;
    }

private:
    int res = 0, size = 0;
    std::vector<float> data;                 // LAYERS x size x size
    std::vector<float> offX, offZ, weight;   // photons
    float builtTime = 0.0f;
    uint64_t builtWorld = 0;
    Vec3 builtSun;

    // Splits 0 .. count-1 into one contiguous range per thread
    template <typename Fn>
    static void runParallel(int count, int threads, Fn fn) {
        threads = std::max(1, std::min(threads, count));
        g_pool.run(threads, [&](int t) {
            int begin = int(int64_t(count) * t / threads);
            int end = int(int64_t(count) * (t + 1) / threads);
            fn(begin, end);
        });
    }

    // Separable Gaussian blur of one layer (sigma in texels)
    void blur(float* layer, float* tmp, float sigma) const {
        if (sigma < 0.3f) return;
        int radius = std::min(12, static_cast<int>(std::ceil(sigma * 2.5f)));
        float kernel[25];
        float sum = 0.0f;
        for (int k = -radius; k <= radius; k++) {
            kernel[k + radius] = std::exp(-0.5f * k * k / (sigma * sigma));
            sum += kernel[k + radius];
        }
        for (int k = 0; k <= 2 * radius; k++) kernel[k] /= sum;
        for (int pass = 0; pass < 2; pass++) {
            const float* src = pass == 0 ? layer : tmp;
            float* dst = pass == 0 ? tmp : layer;
            for (int v = 0; v < size; v++) {
                for (int u = 0; u < size; u++) {
                    // pass 0 blurs along x (u is x), pass 1 along z (u is z)
                    int x = pass == 0 ? u : v, z = pass == 0 ? v : u;
                    float acc = 0.0f;
                    for (int k = -radius; k <= radius; k++) {
                        int q = std::min(size - 1, std::max(0, u + k));
                        int idx = pass == 0 ? z * size + q : q * size + x;
                        acc += src[idx] * kernel[k + radius];
                    }
                    dst[size_t(z) * size + x] = acc;
                }
            }
        }
    }
};

CausticMap g_caustics;

// Caustic light at a point, relative to flat water, with the strength setting applied
inline float causticAt(float x, float z, float depth) {
    if (!g_settings.enableCaustics || !g_caustics.ready()) return 1.0f;
    float c = g_caustics.sample(x, z, depth);
    return std::max(0.0f, 1.0f + (c - 1.0f) * g_settings.causticStrength);
}

// The same per color. Water bends blue slightly more than red, so the three
// colors land a little apart, more so with depth: faint colored fringes.
inline Vec3 causticColor(float x, float z, float depth, const SunLight& sun) {
    float hx = sun.refracted.x, hz = sun.refracted.z;
    float len = std::sqrt(hx * hx + hz * hz);
    if (len < 1e-4f) { hx = 1.0f; hz = 0.0f; len = 1.0f; }
    float shift = 0.012f * depth / len;
    return Vec3(causticAt(x - hx * shift, z - hz * shift, depth),
                causticAt(x, z, depth),
                causticAt(x + hx * shift, z + hz * shift, depth));
}

// Calculate volumetrics: sunlight scattered toward the eye along a ray. The
// shadow rays all point at the sun, so they are marched as SIMD packets
// (raycastShadow8).
//
// fullQuality marches all 12 steps (rays the camera sees directly or through
// water). Otherwise one randomly chosen step is evaluated and scaled by 12: the
// average is the same, at a twelfth of the work, and the extra noise lands on
// indirect light where it is not visible.
//
// In water each step is lit through the caustic map, so the haze breaks into
// shafts that line up with the bright lines on the lake bed, and both the
// light's way down and the way to the eye lose light per color.
Vec3 calculateVolumetrics(const Ray& ray, float maxDist, const World& world, const SunLight& sun, bool inWater,
                          bool fullQuality) {
    if (!g_settings.enableVolumetrics) return Vec3(0, 0, 0);

    const int numSamples = 12;
    const int evaluated = fullQuality ? numSamples : 1;
    const int firstStep = fullQuality ? 0 : std::min(numSamples - 1, int(random01() * numSamples));
    // Water is murkier than air, so its shafts are sampled over a shorter stretch
    float stepSize = std::min(maxDist, inWater ? 20.0f : 50.0f) / float(numSamples);

    // Jittered sample positions along the ray (SoA, padded to 2 groups of 8)
    alignas(32) float ts[16], sx[16], sy[16], sz[16];
    bool laneActive[16];
    for (int i = 0; i < evaluated; i++) {
        float t = stepSize * (firstStep + i + random01() * 0.5f);
        Vec3 samplePos = ray.at(t);
        ts[i] = t;
        sx[i] = samplePos.x;
        sy[i] = samplePos.y;
        sz[i] = samplePos.z;
        bool sampleInWater = (world.getBlock(static_cast<int>(std::floor(samplePos.x)),
                                            static_cast<int>(std::floor(samplePos.y)),
                                            static_cast<int>(std::floor(samplePos.z))) == WATER);
        laneActive[i] = (sampleInWater == inWater);
    }
    const int groups = (evaluated + 7) / 8;
    for (int i = evaluated; i < groups * 8; i++) {      // pad the last group of 8
        ts[i] = 0; sx[i] = 0; sy[i] = 0; sz[i] = 0;
        laneActive[i] = false;
    }

    // How much sun reaches each sample: 1, 0.3 through leaves, 0 behind solid blocks
    Vec3 toSun = -sun.direction;
    alignas(32) float visible[16];
    bool anyVisible = false;
    for (int group = 0; group < groups; group++) {
        int base = group * 8;
        uint8_t hitBlock[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        if (evaluated == 1) {
            // A packet with one live lane costs almost a full packet; use a plain ray.
            if (laneActive[0]) hitBlock[0] = world.shadowBlock(Vec3(sx[0], sy[0], sz[0]), toSun, 100.0f);
        } else {
            world.raycastShadow8(sx + base, sy + base, sz + base, toSun, 100.0f,
                                 laneActive + base, hitBlock);
        }
        for (int L = 0; L < 8; L++) {
            uint8_t hb = hitBlock[L];
            visible[base + L] = !laneActive[base + L] ? 0.0f : hb == AIR ? 1.0f : hb == LEAVES ? 0.3f : 0.0f;
            anyVisible = anyVisible || visible[base + L] > 0.0f;
        }
    }
    // No sample sees the sun: every term below would be multiplied by zero
    if (!anyVisible) return Vec3(0, 0, 0);
    const float scaleUp = float(numSamples) / float(evaluated);

    float cosTheta = ray.direction.dot(inWater ? -sun.refracted : -sun.direction);
    // Water scatters less sharply forward than haze, so shafts also show from the side
    float g = inWater ? 0.5f : 0.6f;
    float phase = (1.0f - g * g) / (4.0f * M_PI * std::pow(1.0f + g * g - 2.0f * g * cosTheta, 1.5f));

    if (inWater) {
        const float invDown = 1.0f / std::max(0.05f, -sun.refracted.y);
        // Sun to each sample (slanted path down), then sample to the eye: the
        // light that survives, per color.
        Vec3 sum(0, 0, 0);
        if (evaluated == 1) {
            float depth = std::max(0.0f, float(WATER_LEVEL) - sy[0]);
            float light = visible[0] * causticAt(sx[0], sz[0], depth);
            sum += waterTransmittance(depth * invDown + ts[0]) * light;
        } else {
            // All samples go through exp8 together, one color at a time
            alignas(32) float tr[3][16];
            float depths[16];
            for (int i = 0; i < groups * 8; i++) {
                depths[i] = std::max(0.0f, float(WATER_LEVEL) - sy[i]);
                float path = visible[i] > 0.0f ? depths[i] * invDown + ts[i] : 0.0f;
                for (int c = 0; c < 3; c++) tr[c][i] = -WATER_EXTINCTION[c] * path;
            }
            for (int c = 0; c < 3; c++) {
                for (int group = 0; group < groups; group++) exp8(F8::load(tr[c] + group * 8)).store(tr[c] + group * 8);
            }
            for (int i = 0; i < evaluated; i++) {
                if (visible[i] <= 0.0f) continue;
                float light = visible[i] * causticAt(sx[i], sz[i], depths[i]);
                sum += Vec3(tr[0][i], tr[1][i], tr[2][i]) * light;
            }
        }
        return sun.getLightContribution() * sum *
               (scaleUp * sun.beamGain * UNDERWATER_SUN_GAIN * WATER_SCATTER * phase * stepSize *
                g_settings.shaftStrength);
    }

    // Air: haze that thins with height
    float total = 0.0f;
    for (int group = 0; group < groups; group++) {
        int base = group * 8;
        F8 li = F8::load(visible + base) * F8(sun.intensity);
        F8 heightFactor = exp8((F8::load(sy + base) - F8(10.0f)) * F8(-0.02f));
        F8 att = min8(F8(1.0f), max8(F8(0.1f), heightFactor));
        F8 absorption = exp8(F8::load(ts + base) * F8(-0.01f));
        total += hsum8(li * att * absorption);
    }
    total *= scaleUp;
    float timeStrength = 1.0f + 2.0f * (1.0f - std::abs(g_settings.timeOfDay - 0.5f) * 2.0f);
    Vec3 volumetricLight = sun.color * (total * 0.04f * phase * stepSize);
    return volumetricLight * Vec3(1.0f, 0.95f, 0.9f) * (timeStrength * 3.0f);
}

// Specks drifting in the water, lit by the sun through the caustic map. Each
// block of water holds a few of them at hashed positions; a ray picks up the
// ones it passes close to. Only the first stretch of a ray is checked, and
// only for a camera that is itself under water.
Vec3 waterParticles(const Ray& ray, float maxDist, const World& world, const SunLight& sun) {
    const float reach = std::min(maxDist, 10.0f);
    const float pixel = 2.0f * std::tan(FOV * float(M_PI) / 360.0f) / float(g_settings.renderHeight);
    const float invDown = 1.0f / std::max(0.05f, -sun.refracted.y);
    const float time = g_settings.waterAnimation;
    const Vec3 o = ray.origin, dir = ray.direction;

    int x = static_cast<int>(std::floor(o.x));
    int y = static_cast<int>(std::floor(o.y));
    int z = static_cast<int>(std::floor(o.z));
    int stepX = dir.x > 0 ? 1 : -1, stepY = dir.y > 0 ? 1 : -1, stepZ = dir.z > 0 ? 1 : -1;
    float tMaxX = (dir.x != 0) ? ((x + (stepX > 0 ? 1 : 0)) - o.x) / dir.x : 1e30f;
    float tMaxY = (dir.y != 0) ? ((y + (stepY > 0 ? 1 : 0)) - o.y) / dir.y : 1e30f;
    float tMaxZ = (dir.z != 0) ? ((z + (stepZ > 0 ? 1 : 0)) - o.z) / dir.z : 1e30f;
    float tDeltaX = (dir.x != 0) ? stepX / dir.x : 1e30f;
    float tDeltaY = (dir.y != 0) ? stepY / dir.y : 1e30f;
    float tDeltaZ = (dir.z != 0) ? stepZ / dir.z : 1e30f;

    Vec3 sum(0, 0, 0);
    float dist = 0.0f;
    while (dist < reach) {
        if (y < WATER_LEVEL && world.getBlock(x, y, z) == WATER) {
            for (uint32_t k = 0; k < 3; k++) {
                uint32_t h = hashCell(x, y, z, 0x51ed270bU + k);
                Vec3 p(x + hashFloat(h), y + hashFloat(hash32(h + 1)), z + hashFloat(hash32(h + 2)));
                Vec3 v = p - o;
                float t = v.dot(dir);
                if (t < 0.6f || t > reach + 0.5f) continue;
                if (v.dot(v) - t * t > 0.09f) continue;                 // not near the ray, even after drifting
                // Slow drift
                float ph = hashFloat(hash32(h + 3)) * 6.2831853f;
                p = p + Vec3(0.12f * std::sin(time * 0.35f + ph), 0.08f * std::sin(time * 0.27f + ph * 1.7f),
                             0.12f * std::cos(time * 0.31f + ph));
                if (p.y > float(WATER_LEVEL) - 0.05f) continue;
                v = p - o;
                t = v.dot(dir);
                if (t < 0.6f || t > reach) continue;
                float perp2 = std::max(0.0f, v.dot(v) - t * t);
                float radius = 0.0015f + 0.0035f * hashFloat(hash32(h + 4));
                float foot = 0.6f * pixel * t;                          // about half a pixel at that distance
                float r2 = radius * radius + foot * foot;
                if (perp2 > 6.0f * r2) continue;
                float depth = float(WATER_LEVEL) - p.y;
                // Smaller than a pixel: spread over the pixel, keeping its total light
                float glint = std::exp(-perp2 / (2.0f * r2)) * (radius * radius / r2);
                sum += waterTransmittance(depth * invDown + t) * (causticAt(p.x, p.z, depth) * glint);
            }
        }
        if (tMaxX < tMaxY && tMaxX < tMaxZ) { x += stepX; dist = tMaxX; tMaxX += tDeltaX; }
        else if (tMaxY < tMaxZ) { y += stepY; dist = tMaxY; tMaxY += tDeltaY; }
        else { z += stepZ; dist = tMaxZ; tMaxZ += tDeltaZ; }
    }
    return sun.getLightContribution() * sum * (sun.beamGain * UNDERWATER_SUN_GAIN * PARTICLE_BRIGHTNESS);
}

// Path tracing
// insideWater: the ray travels through water (it then loses light per color and picks up the water's glow).
// cameraPath: the ray comes from the camera, directly or through water
// refraction/reflection (not after a diffuse bounce). Effects run at full quality on it.
// lampFrom/bouncePdf: set on a diffuse bounce ray. The surface at *lampFrom has
// already sampled the nearby light blocks directly; if this ray then hits one of
// them, its glow is weighted against that sample so the light is not counted twice.
Vec3 trace(const Ray& ray, const World& world, int depth, bool insideWater = false, bool cameraPath = false,
           const Vec3* lampFrom = nullptr, float bouncePdf = 0.0f);

// Light given off by a block (lamps are dimmer in the middle of the day)
inline Vec3 lampEmission(BlockType block) {
    if (block == LIGHT) {
        float brightness = 0.3f + 0.7f * std::abs(g_settings.timeOfDay - 0.5f) * 2.0f;
        return g_materials[LIGHT].emission * brightness;
    }
    return g_materials[block].emission;
}

// Minimal PNG writer: 8-bit RGB, stored (uncompressed) deflate blocks.
// No external library; any viewer or video tool reads the result.
static uint32_t pngCrc(const uint8_t* data, size_t n, uint32_t crc) {
    static uint32_t table[256];
    static bool ready = false;
    if (!ready) {
        for (uint32_t i = 0; i < 256; i++) {
            uint32_t c = i;
            for (int k = 0; k < 8; k++) c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
            table[i] = c;
        }
        ready = true;
    }
    crc = ~crc;
    for (size_t i = 0; i < n; i++) crc = table[(crc ^ data[i]) & 0xFF] ^ (crc >> 8);
    return ~crc;
}

static bool writePNG(const std::string& filename, const uint8_t* rgb, int width, int height) {
    // Scanlines: a filter byte (0 = none) then the row's RGB bytes
    const size_t rowBytes = size_t(width) * 3 + 1;
    std::vector<uint8_t> raw(rowBytes * height);
    for (int y = 0; y < height; y++) {
        raw[y * rowBytes] = 0;
        std::memcpy(&raw[y * rowBytes + 1], rgb + size_t(y) * width * 3, size_t(width) * 3);
    }
    // zlib stream of stored blocks (at most 65535 bytes each) plus Adler-32
    std::vector<uint8_t> z;
    z.reserve(raw.size() + raw.size() / 65535 * 5 + 16);
    z.push_back(0x78);
    z.push_back(0x01);
    uint32_t a = 1, b = 0;
    for (size_t pos = 0; pos < raw.size();) {
        size_t n = std::min<size_t>(65535, raw.size() - pos);
        z.push_back(pos + n == raw.size() ? 1 : 0);
        z.push_back(uint8_t(n & 0xFF));
        z.push_back(uint8_t(n >> 8));
        z.push_back(uint8_t(~n & 0xFF));
        z.push_back(uint8_t((~n >> 8) & 0xFF));
        for (size_t i = 0; i < n; i++) {
            a = (a + raw[pos + i]) % 65521;
            b = (b + a) % 65521;
        }
        z.insert(z.end(), raw.begin() + pos, raw.begin() + pos + n);
        pos += n;
    }
    uint32_t adler = (b << 16) | a;
    for (int s = 24; s >= 0; s -= 8) z.push_back(uint8_t(adler >> s));

    std::ofstream file(filename, std::ios::binary);
    if (!file) return false;
    auto put32 = [&](uint32_t v) {
        const uint8_t bytes[4] = {uint8_t(v >> 24), uint8_t(v >> 16), uint8_t(v >> 8), uint8_t(v)};
        file.write(reinterpret_cast<const char*>(bytes), 4);
    };
    auto chunk = [&](const char* type, const uint8_t* data, size_t n) {
        put32(uint32_t(n));
        file.write(type, 4);
        if (n) file.write(reinterpret_cast<const char*>(data), n);
        uint32_t crc = pngCrc(reinterpret_cast<const uint8_t*>(type), 4, 0);
        if (n) crc = pngCrc(data, n, crc);
        put32(crc);
    };
    const uint8_t signature[8] = {0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};
    file.write(reinterpret_cast<const char*>(signature), 8);
    const uint8_t header[13] = {
        uint8_t(width >> 24), uint8_t(width >> 16), uint8_t(width >> 8), uint8_t(width),
        uint8_t(height >> 24), uint8_t(height >> 16), uint8_t(height >> 8), uint8_t(height),
        8, 2, 0, 0, 0};                           // 8 bits per channel, RGB
    chunk("IHDR", header, 13);
    chunk("IDAT", z.data(), z.size());
    chunk("IEND", nullptr, 0);
    return bool(file);
}

// Renderer - modified to support offline rendering
class Renderer {
    std::vector<uint32_t> framebuffer;
    std::vector<Vec3> accumulator;
    std::atomic<int> nextTile;
    std::atomic<uint64_t> rayCount{0};
    uint64_t frameSeed = 0;
    bool framebufferStale = true;     // the accumulator has passes the framebuffer does not show yet
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
        
        nextTile = 0;
        sampleCount++;

        // Per-pass state shared by all render threads
        g_sun.updateFromTimeOfDay(g_settings.timeOfDay);
        world.prepareSunShadows(g_sun.direction * -1.0f);
        if (g_settings.enableCaustics) {
            static const int texelsPerBlock[3] = {2, 4, 8};
            int quality = g_settings.mode == Settings::MODE_OFFLINE_RENDER ? 3 : g_settings.causticQuality;
            g_caustics.update(world, g_sun, g_settings.waterAnimation, texelsPerBlock[quality - 1]);
        }

        bool cameraUnderwater = getCameraUnderwater(camera, world);

        g_pool.run(renderThreadCount(), [&](int) { renderThread(camera, world, cameraUnderwater); });
        framebufferStale = true;
    }

    // Convert accumulator to framebuffer: tone mapping and gamma, per pixel.
    // Done only when the image is shown or saved (offline rendering and the
    // benchmark need it once per image, not once per pass), on all threads.
    void resolveFramebuffer() {
        if (!framebufferStale) return;
        framebufferStale = false;
        const int total = currentWidth * currentHeight;
        const int threads = std::max(1, std::min(renderThreadCount(), total / 4096));
        g_pool.run(threads, [&](int t) {
            int begin = int(int64_t(total) * t / threads);
            int end = int(int64_t(total) * (t + 1) / threads);
            for (int i = begin; i < end; i++) {
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
        });
    }
    
    // Selects the random sequence for the following passes. With the same seed,
    // settings and pass count, a render is identical from run to run.
    void setFrameSeed(uint64_t seed) { frameSeed = seed; }
    uint64_t getRayCount() const { return rayCount.load(); }
    void resetRayCount() { rayCount = 0; }

    void renderThread(const Camera& camera, const World& world, bool cameraUnderwater) {
        float aspectRatio = float(currentWidth) / currentHeight;
        const uint64_t passSeed = frameSeed * 0x9E3779B97F4A7C15ULL + uint64_t(sampleCount);
        t_rayCount = 0;
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
                    rng.seed(passSeed, uint64_t(y) * currentWidth + x);

                    for (int s = 0; s < SAMPLES_PER_PIXEL; s++) {
                        float u = (x + random01() - currentWidth/2.0f) / (currentWidth/2.0f);
                        float v = -(y + random01() - currentHeight/2.0f) / (currentHeight/2.0f);
                        
                        Ray ray = camera.getRay(u, v, aspectRatio);
                        color = color + trace(ray, world, MAX_BOUNCES, cameraUnderwater, true);
                    }
                    
                    int idx = y * currentWidth + x;
                    accumulator[idx] = accumulator[idx] + color;
                }
            }
        }
        rayCount += t_rayCount;
    }
    
    const uint32_t* getFramebuffer() {
        resolveFramebuffer();
        return framebuffer.data();
    }
    int getSampleCount() const { return sampleCount * SAMPLES_PER_PIXEL; }
    int getWidth() const { return currentWidth; }
    int getHeight() const { return currentHeight; }
    
    bool saveFrame(const std::string& filename) {
        resolveFramebuffer();
        std::vector<uint8_t> rgb(size_t(currentWidth) * currentHeight * 3);
        for (int i = 0; i < currentWidth * currentHeight; i++) {
            uint32_t pixel = framebuffer[i];
            rgb[i * 3 + 0] = (pixel >> 16) & 0xFF;
            rgb[i * 3 + 1] = (pixel >> 8) & 0xFF;
            rgb[i * 3 + 2] = pixel & 0xFF;
        }
        return writePNG(filename, rgb.data(), currentWidth, currentHeight);
    }
};

// Complete trace function implementation
Vec3 trace(const Ray& ray, const World& world, int depth, bool insideWater, bool cameraPath,
           const Vec3* lampFrom, float bouncePdf) {
    if (depth <= 0) return Vec3(0, 0, 0);

    const SunLight& sun = g_sun;

    Vec3 hitPos, hitNormal;
    BlockType hitBlock;

    // Calculate distance to hit (or max distance if no hit)
    float hitDistance = MAX_RAY_DISTANCE;
    bool didHit = world.raycast(ray, MAX_RAY_DISTANCE, hitPos, hitNormal, hitBlock);
    if (didHit) {
        hitDistance = (hitPos - ray.origin).length();
    }

    // Light scattered toward the eye along the ray: haze and shafts, and specks in water
    Vec3 volumetrics(0, 0, 0);
    if (g_settings.enableVolumetrics) {
        volumetrics = calculateVolumetrics(ray, hitDistance, world, sun, insideWater, cameraPath);
    }
    if (insideWater && depth == MAX_BOUNCES && g_settings.enableParticles) {
        volumetrics += waterParticles(ray, hitDistance, world, sun);     // camera under water, its own rays only
    }

    // Every ray ends here: `light` is what arrives at the far end. In water it
    // loses light per color on the way (red first) and the water's own glow
    // takes its place, so distant things fade into blue.
    auto finish = [&](const Vec3& light) {
        if (!insideWater) return light + volumetrics;
        Vec3 through = waterTransmittance(hitDistance);
        float glowY = ray.origin.y + ray.direction.y * std::min(hitDistance, 6.0f) * 0.5f;
        return light * through + waterGlow(glowY, sun) * (Vec3(1, 1, 1) - through) + volumetrics;
    };

    if (!didHit) {
        if (insideWater) return finish(Vec3(0, 0, 0));
        return finish(getSkyColor(ray.direction, g_settings.timeOfDay, sun, cameraPath));
    }

    // Water surface, from either side
    if (hitBlock == WATER) {
        bool fromAbove = !insideWater;
        Vec3 normal = hitNormal;                    // faces the incoming ray

        // Waves on the top surface, seen from above or below
        if (std::abs(hitNormal.y) > 0.9f) {
            normal = getWaterNormal(hitPos, g_settings.waterAnimation);             // points up
            if (!fromAbove) normal = normal * -1.0f;
        }

        float eta = fromAbove ? 1.0f / WATER_IOR : WATER_IOR;
        float cosI = -normal.dot(ray.direction);
        if (cosI < 0.0f) {                          // a wave tilted past the ray: treat as grazing
            normal = normal * -1.0f;
            cosI = -cosI;
        }
        float sinT2 = eta * eta * (1.0f - cosI * cosI);
        bool totalInternalReflection = sinT2 > 1.0f;       // only possible from below

        Vec3 reflectedDir = ray.direction + normal * (2.0f * cosI);
        Vec3 refractedDir = reflectedDir;
        float reflectance = 1.0f;
        if (!totalInternalReflection) {
            float cosT = std::sqrt(1.0f - sinT2);
            refractedDir = ray.direction * eta + normal * (eta * cosI - cosT);
            // Fresnel (Schlick), using the angle on the air side
            float r0 = ((1.0f - WATER_IOR) / (1.0f + WATER_IOR)) * ((1.0f - WATER_IOR) / (1.0f + WATER_IOR));
            reflectance = r0 + (1.0f - r0) * std::pow(1.0f - (fromAbove ? cosI : cosT), 5.0f);
            if (fromAbove) reflectance = std::min(reflectance, 0.8f);  // Reduced max reflectance for more water color
        }

        // A light block reached through the water was already sampled directly
        // by the surface the ray came from (bouncePdf 0 drops its glow here).
        auto traceRefracted = [&]() {
            Ray refractedRay(hitPos + refractedDir * 0.01f, refractedDir);
            return trace(refractedRay, world, depth - 1, fromAbove, cameraPath, lampFrom, 0.0f);
        };
        auto traceReflected = [&]() {
            Ray reflectedRay(hitPos + normal * 0.01f, reflectedDir);
            return trace(reflectedRay, world, depth - 1, insideWater, cameraPath, lampFrom, 0.0f);
        };

        Vec3 result;
        if (totalInternalReflection) {
            result = traceReflected();                      // the mirror around the window to the sky
        } else if (cameraPath) {
            // What the camera sees: both, weighted (the reflection only if it matters)
            result = traceRefracted() * (1.0f - reflectance);
            if (reflectance > 0.02f) result = result + traceReflected() * reflectance;
        } else {
            // Indirect paths follow one of the two, chosen by the reflectance:
            // the same average as tracing both, at half the work.
            result = random01() < reflectance ? traceReflected() : traceRefracted();
        }
        return finish(result);
    }

    const MaterialProps& mat = g_materials[hitBlock];

    // Light blocks: weighted against direct light sampling
    if (isEmitter(hitBlock)) {
        Vec3 emission = lampEmission(hitBlock);
        if (lampFrom) {
            // This ray is a diffuse bounce, and the surface it left also sampled
            // nearby light blocks directly. If this block was one of them, share
            // the result between the two ways of finding it (power heuristic).
            const World::LightCell& cell = world.lightsNear(*lampFrom);
            Vec3 inside = hitPos - hitNormal * 0.5f;
            int lx = static_cast<int>(std::floor(inside.x));
            int ly = static_cast<int>(std::floor(inside.y));
            int lz = static_cast<int>(std::floor(inside.z));
            for (int i = 0; i < cell.count; i++) {
                const Vec3i& L = world.light(cell.index[i]);
                if (L.x == lx && L.y == ly && L.z == lz) {
                    float cosLight = std::max(1e-4f, -hitNormal.dot(ray.direction));
                    float lampPdf = hitDistance * hitDistance / (cosLight * cell.count * 6.0f);
                    emission = emission * (bouncePdf * bouncePdf / (bouncePdf * bouncePdf + lampPdf * lampPdf));
                    break;
                }
            }
        }
        return finish(emission);
    }

    // Does this face touch water? Then it is lit through the water.
    Vec3 outside = hitPos + hitNormal * 0.5f;
    bool wet = world.getBlock(static_cast<int>(std::floor(outside.x)),
                              static_cast<int>(std::floor(outside.y)),
                              static_cast<int>(std::floor(outside.z))) == WATER;

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
        case CORAL_PINK:
        case CORAL_ORANGE:
        case CORAL_PURPLE:
            albedo = getCoralTexture(hitPos, mat.albedo);
            break;
        case KELP:
            albedo = getKelpTexture(hitPos, mat.albedo);
            break;
        default:
            // Use default material albedo for other block types
            break;
    }

    Vec3 shadeOrigin = hitPos + hitNormal * 0.01f;
    Vec3 toSun = sun.direction * -1;
    Vec3 sunLight = sun.getLightContribution() * sun.intensity;
    Vec3 directLight(0, 0, 0);

    if (wet) {
        // Sunlight through the water: it arrives along the refracted direction,
        // focused and spread by the waves (the caustic map), and has lost light
        // per color on its slanted way down. Blocks in the water or above the
        // point where the light entered cast shadows.
        Vec3 up = sun.refracted * -1.0f;
        float cosSun = hitNormal.dot(up);
        if (cosSun > 0.0f) {
            float depthBelow = std::max(0.0f, float(WATER_LEVEL) - hitPos.y);
            float pathLength = depthBelow / std::max(0.05f, up.y);
            Vec3 entry = shadeOrigin + up * pathLength;
            entry.y = float(WATER_LEVEL) + 0.02f;
            bool sunVisible = world.firstSolid(shadeOrigin, up, pathLength) == AIR &&
                              !world.sunOccluded(entry, toSun, 100.0f);
            if (sunVisible) {
                directLight = sunLight * causticColor(hitPos.x, hitPos.z, depthBelow, sun) *
                              waterTransmittance(pathLength) * (sun.beamGain * UNDERWATER_SUN_GAIN * cosSun);
            }
        }
    } else {
        // Direct sun lighting (solid blocks cast shadows; water does not block the sun)
        float sunDot = hitNormal.dot(toSun);
        if (sunDot > 0.0f && !world.sunOccluded(shadeOrigin, toSun, 100.0f)) {
            directLight = sunLight * sunDot;
        }
    }

    // Direct light from nearby light blocks: one random point on one of them
    Vec3 lampLight(0, 0, 0);
    const World::LightCell& cell = world.lightsNear(shadeOrigin);
    if (g_settings.sampleLamps && cell.count > 0) {
        const Vec3i& L = world.light(cell.index[std::min(cell.count - 1, int(random01() * cell.count))]);
        int face = std::min(5, int(random01() * 6.0f));
        int axis = face / 2;
        float side = (face & 1) ? 1.0f : 0.0f;
        float pt[3], faceNormal[3] = {0.0f, 0.0f, 0.0f};
        pt[axis] = side;
        pt[(axis + 1) % 3] = random01();
        pt[(axis + 2) % 3] = random01();
        faceNormal[axis] = side > 0.5f ? 1.0f : -1.0f;
        Vec3 toLamp = Vec3(L.x + pt[0], L.y + pt[1], L.z + pt[2]) - shadeOrigin;
        float dist2 = toLamp.dot(toLamp);
        float dist = std::sqrt(dist2);
        if (dist > 1e-3f) {
            Vec3 wi = toLamp / dist;
            float cosSurface = hitNormal.dot(wi);
            float cosLight = -Vec3(faceNormal[0], faceNormal[1], faceNormal[2]).dot(wi);
            if (cosSurface > 0.0f && cosLight > 0.0f &&
                world.firstSolid(shadeOrigin, wi, dist - 2e-3f) == AIR) {
                // Probability densities (per solid angle) of this direction for
                // the lamp sample and for the diffuse bounce; power heuristic.
                float lampPdf = dist2 / (cosLight * cell.count * 6.0f);
                float bouncePdfHere = cosSurface / float(M_PI);
                float weight = lampPdf * lampPdf / (lampPdf * lampPdf + bouncePdfHere * bouncePdfHere);
                lampLight = lampEmission(world.getBlock(L.x, L.y, L.z)) * (bouncePdfHere / lampPdf * weight);
                if (wet) lampLight = lampLight * waterTransmittance(dist);
            }
        }
    }

    // Indirect lighting: one cosine-weighted bounce. With that distribution the
    // bounce carries the full incoming light, so no fixed ambient term is needed:
    // the sky, other surfaces and (under water) the water's own glow fill the shadows.
    Vec3 bounceDir = randomCosineDirection(hitNormal);
    float cosBounce = std::max(1e-4f, hitNormal.dot(bounceDir));
    Ray scattered(shadeOrigin, bounceDir);
    Vec3 indirectLight = trace(scattered, world, depth - 1, wet, false,
                               g_settings.sampleLamps ? &shadeOrigin : nullptr, cosBounce / float(M_PI));

    // Use the procedurally textured albedo in the final color calculation
    return finish(albedo * (directLight + lampLight + indirectLight));
}

// Fixed benchmark: the same views, sample count and random sequences on every
// machine, so times and images compare across builds and computers. (The
// --benchmark mode plays a camera path in real time, so what it renders depends
// on how fast the machine is.)
struct BenchView {
    const char* name;
    float x, y, z, yaw, pitch;
};
static const BenchView g_benchViews[] = {
    {"lake",       92.5f, 15.3f, 98.2f, 4.19f, -0.20f},
    {"shore",      49.6f, 14.1f, 65.3f, 1.93f, -0.01f},
    {"underwater", 83.8f, 4.83f, 50.99f, 0.997f, 0.078f},
    {"lakebed",    74.5f, 17.0f, 90.5f, 0.80f, -0.90f},
    {"aerial",     64.0f, 75.0f, 64.0f, 0.60f, -1.20f},
    {"deep",       66.5f,  5.5f, 52.5f, 1.571f, -0.12f},
    {"lookup",     65.01f, 8.09f, 59.55f, 4.769f, 0.677f},
};

int runFixedBenchmark(const World& world, int samples) {
    Renderer renderer;
    Camera camera;
    SystemInfo sys = SystemInfo::get();
    int threads = renderThreadCount();
    std::filesystem::create_directories(g_settings.outputDir);

    std::cout << "Fixed benchmark: " << g_settings.renderWidth << "x" << g_settings.renderHeight
              << ", " << samples << " samples/pixel, " << threads << " threads\n";
    std::cout << "CPU: " << sys.cpuModel << "\n\n";
    std::cout << std::left << std::setw(12) << "View" << std::right << std::setw(10) << "Time (s)"
              << std::setw(12) << "Mrays/s" << std::setw(14) << "Msamples/s" << "\n";

    JSONWriter json;
    json.startObject();
    json.startObject("system_info");
    sys.toJSON(json);
    json.endObject();
    json.addNumber("render_width", g_settings.renderWidth);
    json.addNumber("render_height", g_settings.renderHeight);
    json.addNumber("samples_per_pixel", samples);
    json.addNumber("threads", threads);
    json.addBool("caustics", g_settings.enableCaustics);
    json.addBool("volumetrics", g_settings.enableVolumetrics);
    json.startArray("views");

    double totalTime = 0;
    uint64_t totalRays = 0;
    int viewIndex = 0;
    for (const BenchView& v : g_benchViews) {
        camera.setFromKeyframe(v.x, v.y, v.z, v.yaw, v.pitch);
        g_settings.waterAnimation = 2.0f;
        renderer.reset();
        renderer.resetRayCount();
        renderer.setFrameSeed(uint64_t(viewIndex++));

        auto t0 = std::chrono::high_resolution_clock::now();
        while (renderer.getSampleCount() < samples) renderer.render(camera, world, false);
        double seconds = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();

        uint64_t rays = renderer.getRayCount();
        double pixelSamples = double(g_settings.renderWidth) * g_settings.renderHeight * renderer.getSampleCount();
        std::string image = g_settings.outputDir + "/bench_" + v.name + ".png";
        renderer.saveFrame(image);
        std::cout << std::left << std::setw(12) << v.name << std::right << std::fixed
                  << std::setw(10) << std::setprecision(2) << seconds
                  << std::setw(12) << std::setprecision(1) << rays / seconds / 1e6
                  << std::setw(14) << std::setprecision(2) << pixelSamples / seconds / 1e6 << "\n";

        json.startObject();
        json.addString("name", v.name);
        json.addNumber("seconds", seconds);
        json.addNumber("rays", double(rays));
        json.addNumber("mrays_per_second", rays / seconds / 1e6);
        json.addNumber("msamples_per_second", pixelSamples / seconds / 1e6);
        json.addString("image", image);
        json.endObject();
        totalTime += seconds;
        totalRays += rays;
    }
    json.endArray();
    json.addNumber("total_seconds", totalTime);
    json.addNumber("total_mrays_per_second", totalRays / totalTime / 1e6);
    json.endObject();

    std::cout << std::left << std::setw(12) << "total" << std::right << std::fixed
              << std::setw(10) << std::setprecision(2) << totalTime
              << std::setw(12) << std::setprecision(1) << totalRays / totalTime / 1e6 << "\n";
    std::ofstream file("benchmark_fixed.json");
    file << json.toString();
    std::cout << "\nImages: " << g_settings.outputDir << "/bench_<view>.png   Results: benchmark_fixed.json\n";
    return 0;
}

// Main function with demo recording
int main(int argc, char* argv[]) {
    // Parse command line arguments
    bool offlineMode = false;
    bool benchmarkMode = false;
    bool fixedBenchmark = false;
    bool dumpCaustics = false;
    bool samplesGiven = false;
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
        } else if (arg == "--bench") {
            fixedBenchmark = true;
        } else if (arg == "--samples" && i + 1 < argc) {
            g_settings.offlineTargetSamples = std::max(1, std::stoi(argv[++i]));
            samplesGiven = true;
        } else if (arg == "--threads" && i + 1 < argc) {
            g_settings.threads = std::max(0, std::stoi(argv[++i]));
        } else if (arg == "--seed" && i + 1 < argc) {
            g_settings.worldSeed = std::stoi(argv[++i]);
        } else if (arg == "--no-caustics") {
            g_settings.enableCaustics = false;
        } else if (arg == "--no-volumetrics") {
            g_settings.enableVolumetrics = false;
        } else if (arg == "--no-lamp-sampling") {
            g_settings.sampleLamps = false;
        } else if (arg == "--dump-caustics") {
            dumpCaustics = true;
        } else if (arg == "--no-particles") {
            g_settings.enableParticles = false;
        } else if (arg == "--caustic-strength" && i + 1 < argc) {
            g_settings.causticStrength = std::max(0.0f, std::stof(argv[++i]));
        } else if (arg == "--shaft-strength" && i + 1 < argc) {
            g_settings.shaftStrength = std::max(0.0f, std::stof(argv[++i]));
        } else if (arg == "--time" && i + 1 < argc) {
            g_settings.timeOfDay = std::max(0.0f, std::min(1.0f, std::stof(argv[++i])));
        } else if (arg == "--resolution" && i + 1 < argc) {
            int preset = std::stoi(argv[++i]);
            g_settings.adjustRenderResolution(preset);
        } else if (arg == "--caustic-quality" && i + 1 < argc) {
            g_settings.causticQuality = std::max(1, std::min(3, std::stoi(argv[++i])));
        } else if (arg == "--play") {
            g_settings.mode = Settings::MODE_PLAYBACK;
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: pathtracer [demo.json] [options]\n"
                         "  demo.json            camera path to load (same as --demo)\n"
                         "  --demo <file>        camera path file (default demo.json)\n"
                         "  --play               play the camera path in the window\n"
                         "  --benchmark          play the camera path and write benchmark_results.json\n"
                         "  --bench              render fixed views, print timings, write benchmark_fixed.json (no window)\n"
                         "  --offline            render the camera path to output/frame_NNNNN.png (no window)\n"
                         "  --samples <n>        samples per pixel: offline frames (default 1000), --bench (default 32)\n"
                         "  --resolution <1-6>   144p, 240p, 360p (default), 480p, 720p, 1080p\n"
                         "  --threads <n>        render threads (default: all)\n"
                         "  --seed <n>           world seed (default 42)\n"
                         "  --caustic-quality <1-3>  caustic map detail: 2, 4 or 8 texels per block (default 2; offline uses 3)\n"
                         "  --caustic-strength <x>   contrast of the caustic pattern (default 1, 0 = even light)\n"
                         "  --shaft-strength <x>     brightness of underwater light shafts (default 1)\n"
                         "  --no-particles           no drifting specks in the water\n"
                         "  --dump-caustics          write the caustic map's layers to output/ as images and exit\n"
                         "  --time <0-1>         time of day (default 0.85; 0.5 is midday)\n"
                         "  --no-caustics, --no-volumetrics   turn an effect off\n"
                         "  --no-lamp-sampling   find light blocks by bounces only (slower to converge; for comparison)\n";
            return 0;
        } else if (!arg.empty() && arg[0] != '-') {
            demoFile = arg;
        } else {
            std::cerr << "Unknown or incomplete option: " << arg << " (see --help)\n";
            return 1;
        }
    }
    
    if (dumpCaustics) {
        // Debug aid: the caustic map's layers as images (white = 3x the light under flat water)
        World dumpWorld;
        dumpWorld.generate(g_settings.worldSeed);
        g_sun.updateFromTimeOfDay(g_settings.timeOfDay);
        static const int texelsPerBlock[3] = {2, 4, 8};
        g_caustics.update(dumpWorld, g_sun, 2.0f, texelsPerBlock[g_settings.causticQuality - 1]);
        std::filesystem::create_directories(g_settings.outputDir);
        int n = g_caustics.texels();
        std::vector<uint8_t> rgb(size_t(n) * n * 3);
        for (int d = 0; d < CausticMap::LAYERS; d++) {
            const float* layer = g_caustics.layer(d);
            for (int i = 0; i < n * n; i++) {
                uint8_t v = uint8_t(std::min(255.0f, std::max(0.0f, layer[i] * 85.0f)));
                rgb[i * 3] = rgb[i * 3 + 1] = rgb[i * 3 + 2] = v;
            }
            std::stringstream ss;
            ss << g_settings.outputDir << "/caustics_depth_" << std::setfill('0') << std::setw(2) << d << ".png";
            writePNG(ss.str(), rgb.data(), n, n);
        }
        std::cout << "Wrote " << CausticMap::LAYERS << " caustic layers to " << g_settings.outputDir << "/\n";
        return 0;
    }

    if (fixedBenchmark) {
        World benchWorld;
        benchWorld.generate(g_settings.worldSeed);
        return runFixedBenchmark(benchWorld, samplesGiven ? g_settings.offlineTargetSamples : 32);
    }

    // Offline rendering needs no window, so it also runs on machines without a display.
    if (!offlineMode && SDL_Init(SDL_INIT_VIDEO) < 0) {
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
    const Uint8* keystate = offlineMode ? nullptr : SDL_GetKeyboardState(nullptr);
    
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
        std::filesystem::create_directories(g_settings.outputDir);
    }

    int offlineFrameCount = 0;
    uint64_t renderedFrames = 0;
    
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
                                case 1: std::cout << "Low (2 texels per block)\n"; break;
                                case 2: std::cout << "Medium (4 texels per block)\n"; break;
                                case 3: std::cout << "High (8 texels per block)\n"; break;
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
        // Re-read the clock here: F1 resets startTime during event handling above.
        totalElapsed = std::chrono::duration<float>(currentTime - startTime).count();
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
                // The water follows the frame's time, so it holds still while a
                // frame accumulates and its speed does not depend on --samples.
                g_settings.waterAnimation = demoTime * WATER_ANIM_SPEED;
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
        
        // Water animates in real time while the view is changing. While the
        // camera is still the image accumulates, so the water holds its pose
        // (accumulating over moving waves would blur the caustics forever).
        if (g_settings.mode != Settings::MODE_OFFLINE_RENDER && (cameraMoving || needsReset)) {
            g_settings.waterAnimation += std::min(deltaTime, 0.1f) * WATER_ANIM_SPEED;
        }
        
        prevCamera = camera;
        
        // Render. Offline frames are seeded by their frame number, so re-rendering
        // a frame reproduces it exactly; the live view just needs fresh noise.
        renderer.setFrameSeed(g_settings.mode == Settings::MODE_OFFLINE_RENDER ? uint64_t(offlineFrameCount)
                                                                              : renderedFrames++);
        auto renderStart = std::chrono::high_resolution_clock::now();
        renderer.render(camera, world, cameraMoving || needsReset);
        auto renderEnd = std::chrono::high_resolution_clock::now();
        float renderTime = std::chrono::duration<float, std::milli>(renderEnd - renderStart).count();
        
        // Save frame for offline rendering
        if (g_settings.mode == Settings::MODE_OFFLINE_RENDER) {
            if (renderer.getSampleCount() >= g_settings.offlineTargetSamples) {
                std::stringstream ss;
                ss << g_settings.outputDir << "/frame_" << std::setfill('0') 
                   << std::setw(5) << offlineFrameCount << ".png";
                if (!renderer.saveFrame(ss.str())) {
                    std::cerr << "Could not write " << ss.str() << "\n";
                    return 1;
                }
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
