// Copyright (c) 2016 Doyub Kim

#include <pch.h>
#include <jet/logging.h>
#include <jet/macros.h>

#include <chrono>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iostream>
#include <mutex>
#include <string>

namespace jet {

static std::mutex critical;

static std::ostream* infoOutStream = &std::cout;
static std::ostream* warnOutStream = &std::cout;
static std::ostream* errorOutStream = &std::cerr;
static std::ostream* debugOutStream = &std::cout;

inline std::ostream* levelToStream(LoggingLevel level) {
    switch (level) {
        case LoggingLevel::Info:
            return infoOutStream;
        case LoggingLevel::Warn:
            return warnOutStream;
        case LoggingLevel::Error:
            return errorOutStream;
        case LoggingLevel::Debug:
            return debugOutStream;
    }
    return nullptr;
}

inline std::string levelToString(LoggingLevel level) {
    switch (level) {
        case LoggingLevel::Info:
            return "INFO";
        case LoggingLevel::Warn:
            return "WARN";
        case LoggingLevel::Error:
            return "ERROR";
        case LoggingLevel::Debug:
            return "DEBUG";
    }
    return nullptr;
}


Logger::Logger(LoggingLevel level)
    : _level(level) {
}

Logger::~Logger() {
    std::lock_guard<std::mutex> lock(critical);
#ifdef JET_DEBUG_MODE
    if (_level != LoggingLevel::Debug) {
#endif
    auto strm = levelToStream(_level);
    (*strm) << _buffer.str() << std::endl;
    strm->flush();
#ifdef JET_DEBUG_MODE
    }
#endif
}


void Logging::setInfoStream(std::ostream* strm) {
    std::lock_guard<std::mutex> lock(critical);
    infoOutStream = strm;
}

void Logging::setWarnStream(std::ostream* strm) {
    std::lock_guard<std::mutex> lock(critical);
    warnOutStream = strm;
}

void Logging::setErrorStream(std::ostream* strm) {
    std::lock_guard<std::mutex> lock(critical);
    errorOutStream = strm;
}

void Logging::setDebugStream(std::ostream* strm) {
    std::lock_guard<std::mutex> lock(critical);
    debugOutStream = strm;
}

void Logging::setAllStream(std::ostream* strm) {
    setInfoStream(strm);
    setWarnStream(strm);
    setErrorStream(strm);
    setDebugStream(strm);
}

std::string Logging::getHeader(LoggingLevel level) {
    auto now = std::chrono::system_clock::to_time_t(
        std::chrono::system_clock::now());
    char timeStr[20];
#ifdef JET_WINDOWS
    tm time;
    localtime_s(&time, &now);
    strftime(timeStr, sizeof(timeStr), "%F %T", &time);
#else
    strftime(timeStr, sizeof(timeStr), "%F %T", std::localtime(&now));
#endif
    char header[256];
    snprintf(
        header, sizeof(header), "[%s] %s ",
        levelToString(level).c_str(),
        timeStr);
    return header;
}

}  // namespace jet
