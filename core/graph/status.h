#ifndef CORE_GRAPH_STATUS_H
#define CORE_GRAPH_STATUS_H

#include <memory>
#include <string>


namespace Lotus
{
    namespace Common
    {

#define RETURN_IF_ERROR(expr)           \
  do {                                  \
    auto status = (expr);               \
    if ((!status.Ok())) return status;  \
  } while (0)

        typedef enum {
            OK = 0,
            FAIL = 1,
            INVALID_ARGUMENT = 2,
            NO_SUCHFILE = 3,
            NO_MODEL = 4,
            ENGINE_ERROR = 5,
            RUNTIME_EXCEPTION = 6,
            INVALID_PROTOBUF = 7,
            MODEL_LOADED = 8,
            NOT_IMPLEMENTED = 9,
        } StatusCode;


        class Status
        {
        public:

            Status() {}

            Status(StatusCode code, const std::string& msg);

            explicit Status(StatusCode code);

            inline Status(const Status& s)
                : m_state((s.m_state == NULL) ? NULL : new State(*s.m_state)) {}

            bool Ok() const;

            StatusCode Code() const;

            const std::string& ErrorMessage() const;

            std::string ToString() const;

            inline void operator=(const Status& s)
            {
                if (nullptr == s.m_state)
                {
                    m_state.reset();
                }
                else if (m_state != s.m_state)
                {
                    m_state.reset(new State(*s.m_state));
                }
            }

            inline bool operator==(const Status& x) const
            {
                return (this->m_state == x.m_state) || (ToString() == x.ToString());
            }

            inline bool operator!=(const Status& x) const
            {
                return !(*this == x);
            }

            static Status OK();

        private:

            static const std::string& EmptyString();

            struct State
            {
                StatusCode m_code;
                std::string m_msg;
            };

            // As long as Code() is OK, m_state == NULL.
            std::unique_ptr<State> m_state;
        };
    }
}

#endif // !CORE_GRAPH_STATUS_H
