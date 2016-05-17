// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_LOGGING_H_
#define INCLUDE_JET_LOGGING_H_

#include <iostream>
#include <sstream>
#include <string>

namespace jet {

enum class LoggingLevel {
    Info,
    Warn,
    Error,
    Debug
};

//!
//! \brief Super simple logger implementation.
//!
//! This is a super simple logger implementation that has minimal logging
//! capability. Currently, the class doesn't support multi-thread logging.
//!
class Logger final {
 public:
    explicit Logger(LoggingLevel level);

    ~Logger();

    template <typename T>
    const Logger& operator<<(const T& x) const {
        _buffer << x;
        return *this;
    }

 private:
    LoggingLevel _level;
    mutable std::stringstream _buffer;
};

class Logging {
 public:
    static void setInfoStream(std::ostream* strm);

    static void setWarnStream(std::ostream* strm);

    static void setErrorStream(std::ostream* strm);

    static void setDebugStream(std::ostream* strm);

    static void setAllStream(std::ostream* strm);

    static std::string getHeader(LoggingLevel level);
};

extern Logger infoLogger;
extern Logger warnLogger;
extern Logger errorLogger;
extern Logger debugLogger;

#define JET_INFO \
    (Logger(LoggingLevel::Info) << Logging::getHeader(LoggingLevel::Info) \
     << "[" << __FILE__ << ":" << __LINE__ << " (" << __func__ << ")] ")
#define JET_WARN \
    (Logger(LoggingLevel::Warn) << Logging::getHeader(LoggingLevel::Warn) \
     << "[" << __FILE__ << ":" << __LINE__ << " (" << __func__ << ")] ")
#define JET_ERROR \
    (Logger(LoggingLevel::Error) << Logging::getHeader(LoggingLevel::Error) \
     << "[" << __FILE__ << ":" << __LINE__ << " (" << __func__ << ")] ")
#define JET_DEBUG \
    (Logger(LoggingLevel::Debug) << Logging::getHeader(LoggingLevel::Debug) \
     << "[" << __FILE__ << ":" << __LINE__ << " (" << __func__ << ")] ")

}  // namespace jet

#endif  // INCLUDE_JET_LOGGING_H_
