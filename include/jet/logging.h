// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_LOGGING_H_
#define INCLUDE_JET_LOGGING_H_

#include <iostream>
#include <sstream>
#include <string>

namespace jet {

//! Level of the logging.
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
    //! Constructs a logger with logging level.
    explicit Logger(LoggingLevel level);

    //! Destructor.
    ~Logger();

    //! Writes a value to the buffer stream.
    template <typename T>
    const Logger& operator<<(const T& x) const {
        _buffer << x;
        return *this;
    }

 private:
    LoggingLevel _level;
    mutable std::stringstream _buffer;
};

//! Helper class for logging.
class Logging {
 public:
    //! Sets the output stream for the info level logs.
    static void setInfoStream(std::ostream* strm);

    //! Sets the output stream for the warning level logs.
    static void setWarnStream(std::ostream* strm);

    //! Sets the output stream for the error level logs.
    static void setErrorStream(std::ostream* strm);

    //! Sets the output stream for the debug level logs.
    static void setDebugStream(std::ostream* strm);

    //! Sets the output stream for all the log levelss.
    static void setAllStream(std::ostream* strm);

    //! Returns the header string.
    static std::string getHeader(LoggingLevel level);
};

//! Info-level logger.
extern Logger infoLogger;

//! Warn-level logger.
extern Logger warnLogger;

//! Error-level logger.
extern Logger errorLogger;

//! Debug-level logger.
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
