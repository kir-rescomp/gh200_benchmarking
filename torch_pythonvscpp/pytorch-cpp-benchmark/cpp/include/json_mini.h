// Minimal JSON reader for reading config/experiment.json in the harness.
// Not a general-purpose library: supports objects, arrays, strings, numbers,
// booleans, null -- enough for the config file. Header-only, no dependency.
//
// If you would rather use a real library, replace this with nlohmann/json
// and adjust load_config in harness.cpp. Kept dependency-free so the C++
// side builds with nothing but libtorch.
#pragma once

#include <cctype>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

struct JsonValue {
    enum Type { Null, Bool, Number, String, Array, Object } type = Null;
    bool b = false;
    double num = 0;
    std::string str;
    std::vector<JsonValue> arr;
    std::map<std::string, JsonValue> obj;

    bool has(const std::string& k) const {
        return type == Object && obj.count(k);
    }
    const JsonValue& operator[](const std::string& k) const {
        static JsonValue null_v;
        auto it = obj.find(k);
        if (it == obj.end())
            throw std::runtime_error("json key not found: " + k);
        return it->second;
    }
    long as_int() const { return static_cast<long>(num); }
    double as_double() const { return num; }
    bool as_bool() const { return b; }
    const std::string& as_string() const { return str; }
};

class JsonParser {
public:
    explicit JsonParser(const std::string& s) : s_(s), i_(0) {}
    JsonValue parse() { skip(); return value(); }

private:
    const std::string& s_;
    size_t i_;

    void skip() { while (i_ < s_.size() && std::isspace((unsigned char)s_[i_])) ++i_; }
    char peek() { return i_ < s_.size() ? s_[i_] : '\0'; }
    char get() { return s_[i_++]; }

    JsonValue value() {
        skip();
        char c = peek();
        if (c == '{') return object();
        if (c == '[') return array();
        if (c == '"') { JsonValue v; v.type = JsonValue::String; v.str = string(); return v; }
        if (c == 't' || c == 'f') return boolean();
        if (c == 'n') { i_ += 4; JsonValue v; v.type = JsonValue::Null; return v; }
        return number();
    }

    JsonValue object() {
        JsonValue v; v.type = JsonValue::Object;
        get();  // {
        skip();
        if (peek() == '}') { get(); return v; }
        while (true) {
            skip();
            std::string key = string();
            skip(); get();  // :
            v.obj[key] = value();
            skip();
            char c = get();
            if (c == '}') break;
            // c == ','
        }
        return v;
    }

    JsonValue array() {
        JsonValue v; v.type = JsonValue::Array;
        get();  // [
        skip();
        if (peek() == ']') { get(); return v; }
        while (true) {
            v.arr.push_back(value());
            skip();
            char c = get();
            if (c == ']') break;
        }
        return v;
    }

    std::string string() {
        std::string out;
        get();  // opening quote
        while (true) {
            char c = get();
            if (c == '"') break;
            if (c == '\\') {
                char e = get();
                switch (e) {
                    case 'n': out += '\n'; break;
                    case 't': out += '\t'; break;
                    case '"': out += '"'; break;
                    case '\\': out += '\\'; break;
                    case '/': out += '/'; break;
                    default: out += e; break;
                }
            } else {
                out += c;
            }
        }
        return out;
    }

    JsonValue boolean() {
        JsonValue v; v.type = JsonValue::Bool;
        if (peek() == 't') { i_ += 4; v.b = true; }
        else { i_ += 5; v.b = false; }
        return v;
    }

    JsonValue number() {
        size_t start = i_;
        while (i_ < s_.size() &&
               (std::isdigit((unsigned char)s_[i_]) || s_[i_] == '-' ||
                s_[i_] == '+' || s_[i_] == '.' || s_[i_] == 'e' ||
                s_[i_] == 'E'))
            ++i_;
        JsonValue v; v.type = JsonValue::Number;
        v.num = std::stod(s_.substr(start, i_ - start));
        return v;
    }
};

inline JsonValue json_parse_file(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open config: " + path);
    std::stringstream ss;
    ss << f.rdbuf();
    std::string content = ss.str();
    JsonParser p(content);
    return p.parse();
}
