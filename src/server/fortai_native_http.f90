module fortai_native_http
    !! OpenAI-compatible protocol policy for the native FortAI service.
    !!
    !! The C companion only reads and writes socket bytes.  JSON, chat
    !! templating, response serialization, and the web UI live here so they
    !! share the same growable string_t implementation as the runtime.
    use, intrinsic :: iso_c_binding, only: c_char, c_int, c_null_char
    use, intrinsic :: iso_fortran_env, only: int32, int64, real32
    use fortai_native_service, only: fortai_native_service_complete_text_options
    use fortai_string, only: string_t
    implicit none
    private

    integer, parameter :: max_generation = 32768
    integer, parameter :: max_messages = 256

    type :: message_t
        type(string_t) :: role
        type(string_t) :: content
        type(string_t) :: reasoning_content
        type(string_t) :: tool_calls
    end type message_t

    type :: tool_call_t
        type(string_t) :: name
        type(string_t) :: arguments
    end type tool_call_t

    public :: fortai_native_http_handle

contains

    logical function finite_real32(value)
        real(real32), intent(in) :: value
        integer(int32) :: bits
        integer(int32), parameter :: exponent_mask = int(z'7f800000', int32)
        bits = transfer(value, bits)
        finite_real32 = iand(bits, exponent_mask) /= exponent_mask
    end function finite_real32

    function from_c(input, length) result(text)
        character(kind=c_char), intent(in) :: input(*)
        integer, intent(in), optional :: length
        type(string_t) :: text
        call text%from_c(input, length)
    end function from_c

    integer function skip_space(text, position)
        character(len=*), intent(in) :: text
        integer, intent(in) :: position
        skip_space = max(1, position)
        do while (skip_space <= len(text))
            if (text(skip_space:skip_space) /= ' ' .and. text(skip_space:skip_space) /= char(9) .and. &
                text(skip_space:skip_space) /= char(10) .and. text(skip_space:skip_space) /= char(13)) exit
            skip_space = skip_space + 1
        end do
    end function skip_space

    integer function json_key(text, key, first, last)
        character(len=*), intent(in) :: text, key
        integer, intent(in) :: first
        integer, intent(in), optional :: last
        integer :: limit, candidate, colon, offset, found
        character(len=:), allocatable :: needle

        json_key = 0
        limit = len(text)
        if (present(last)) limit = min(limit, last)
        if (first > limit) return
        needle = '"' // key // '"'
        offset = first
        do while (offset <= limit)
            found = index(text(offset:limit), needle)
            if (found == 0) return
            candidate = offset + found - 1
            colon = skip_space(text, candidate + len(needle))
            if (colon > limit) return
            if (colon <= limit) then
                if (text(colon:colon) /= ':') then
                    offset = candidate + len(needle)
                    cycle
                end if
                json_key = skip_space(text, colon + 1)
                return
            end if
            offset = candidate + len(needle)
        end do
    end function json_key

    integer function matching_delimiter(text, first, opening, closing)
        character(len=*), intent(in) :: text
        integer, intent(in) :: first
        character(len=1), intent(in) :: opening, closing
        integer :: position, depth
        logical :: quoted, escaped

        matching_delimiter = 0
        if (first < 1 .or. first > len(text)) return
        if (text(first:first) /= opening) return
        depth = 0
        quoted = .false.
        escaped = .false.
        do position = first, len(text)
            if (quoted) then
                if (escaped) then
                    escaped = .false.
                else if (text(position:position) == '\') then
                    escaped = .true.
                else if (text(position:position) == '"') then
                    quoted = .false.
                end if
            else if (text(position:position) == '"') then
                quoted = .true.
            else if (text(position:position) == opening) then
                depth = depth + 1
            else if (text(position:position) == closing) then
                depth = depth - 1
                if (depth == 0) then
                    matching_delimiter = position
                    return
                end if
            end if
        end do
    end function matching_delimiter

    integer function hex_digit(value)
        character(len=1), intent(in) :: value
        select case (value)
        case ('0':'9'); hex_digit = iachar(value) - iachar('0')
        case ('a':'f'); hex_digit = iachar(value) - iachar('a') + 10
        case ('A':'F'); hex_digit = iachar(value) - iachar('A') + 10
        case default; hex_digit = -1
        end select
    end function hex_digit

    subroutine append_codepoint(output, codepoint)
        type(string_t), intent(inout) :: output
        integer, intent(in) :: codepoint
        integer :: value
        character(len=4) :: bytes

        value = codepoint
        if (value < 0 .or. value > int(z'10ffff')) return
        if (value <= int(z'7f')) then
            call output%append_char(achar(value))
        else if (value <= int(z'7ff')) then
            bytes(1:1) = achar(int(z'c0') + ishft(value, -6))
            bytes(2:2) = achar(int(z'80') + iand(value, int(z'3f')))
            call output%append(bytes(:2))
        else if (value <= int(z'ffff')) then
            bytes(1:1) = achar(int(z'e0') + ishft(value, -12))
            bytes(2:2) = achar(int(z'80') + iand(ishft(value, -6), int(z'3f')))
            bytes(3:3) = achar(int(z'80') + iand(value, int(z'3f')))
            call output%append(bytes(:3))
        else
            bytes(1:1) = achar(int(z'f0') + ishft(value, -18))
            bytes(2:2) = achar(int(z'80') + iand(ishft(value, -12), int(z'3f')))
            bytes(3:3) = achar(int(z'80') + iand(ishft(value, -6), int(z'3f')))
            bytes(4:4) = achar(int(z'80') + iand(value, int(z'3f')))
            call output%append(bytes)
        end if
    end subroutine append_codepoint

    logical function json_string(text, first, output, after)
        character(len=*), intent(in) :: text
        integer, intent(in) :: first
        type(string_t), intent(out) :: output
        integer, intent(out) :: after
        integer :: position, digit, codepoint, high, low
        character(len=1) :: escaped

        call output%clear()
        json_string = .false.
        after = 0
        if (first < 1 .or. first > len(text)) return
        if (text(first:first) /= '"') return
        position = first + 1
        do while (position <= len(text))
            if (text(position:position) == '"') then
                after = position + 1
                json_string = .true.
                return
            else if (text(position:position) /= '\') then
                if (iachar(text(position:position)) < 32) return
                call output%append_char(text(position:position))
                position = position + 1
                cycle
            end if
            position = position + 1
            if (position > len(text)) return
            escaped = text(position:position)
            select case (escaped)
            case ('"', '\', '/')
                call output%append_char(escaped)
                position = position + 1
            case ('b')
                call output%append_char(char(8)); position = position + 1
            case ('f')
                call output%append_char(char(12)); position = position + 1
            case ('n')
                call output%append_char(char(10)); position = position + 1
            case ('r')
                call output%append_char(char(13)); position = position + 1
            case ('t')
                call output%append_char(char(9)); position = position + 1
            case ('u')
                if (position + 4 > len(text)) return
                codepoint = 0
                do digit = 1, 4
                    high = hex_digit(text(position + digit:position + digit))
                    if (high < 0) return
                    codepoint = 16 * codepoint + high
                end do
                position = position + 5
                if (codepoint >= int(z'd800') .and. codepoint <= int(z'dbff') .and. position + 5 <= len(text)) then
                    if (text(position:position + 1) /= '\u') then
                        call append_codepoint(output, codepoint)
                        cycle
                    end if
                    low = 0
                    do digit = 1, 4
                        high = hex_digit(text(position + 1 + digit:position + 1 + digit))
                        if (high < 0) return
                        low = 16 * low + high
                    end do
                    if (low >= int(z'dc00') .and. low <= int(z'dfff')) then
                        codepoint = int(z'10000') + ishft(codepoint - int(z'd800'), 10) + low - int(z'dc00')
                        position = position + 6
                    end if
                end if
                call append_codepoint(output, codepoint)
            case default
                return
            end select
        end do
    end function json_string

    logical function json_content(text, first, output, after)
        character(len=*), intent(in) :: text
        integer, intent(in) :: first
        type(string_t), intent(out) :: output
        integer, intent(out) :: after
        integer :: limit, value, position
        logical :: found
        type(string_t) :: parsed

        call output%clear()
        after = 0
        if (first < 1 .or. first > len(text)) then
            json_content = .false.
            return
        end if
        if (text(first:first) == '"') then
            json_content = json_string(text, first, output, after)
            return
        end if
        if (first + 3 <= len(text)) then
            if (text(first:first + 3) == 'null') then
                call output%clear()
                after = first + 4
                json_content = .true.
                return
            end if
        end if
        if (text(first:first) /= '[' .and. text(first:first) /= '{') then
            json_content = .false.
            return
        end if
        limit = matching_delimiter(text, first, text(first:first), merge(']', '}', text(first:first) == '['))
        if (limit == 0) then
            json_content = .false.
            return
        end if
        call output%clear()
        found = .false.
        position = first + 1
        do
            value = json_key(text, 'text', position, limit)
            if (value == 0) exit
            if (value > limit) exit
            if (text(value:value) /= '"') then
                json_content = .false.
                return
            end if
            if (.not. json_string(text, value, parsed, after)) then
                json_content = .false.
                return
            end if
            call output%append_string(parsed)
            found = .true.
            position = after
            if (position > limit) exit
        end do
        json_content = found
    end function json_content

    logical function json_raw_value(text, first, output, after)
        character(len=*), intent(in) :: text
        integer, intent(in) :: first
        type(string_t), intent(out) :: output
        integer, intent(out) :: after
        integer :: limit, value_end
        type(string_t) :: ignored

        call output%clear()
        after = 0
        json_raw_value = .false.
        if (first < 1 .or. first > len(text)) return
        select case (text(first:first))
        case ('"')
            if (.not. json_string(text, first, ignored, after)) return
            call output%set(text(first:after - 1))
        case ('[', '{')
            limit = matching_delimiter(text, first, text(first:first), merge(']', '}', text(first:first) == '['))
            if (limit == 0) return
            call output%set(text(first:limit))
            after = limit + 1
        case default
            value_end = first
            do while (value_end <= len(text))
                select case (text(value_end:value_end))
                case (',', '}', ']', ' ', char(9), char(10), char(13))
                    exit
                case default
                    value_end = value_end + 1
                end select
            end do
            if (value_end <= first) return
            call output%set(text(first:value_end - 1))
            after = value_end
        end select
        json_raw_value = output%length() > 0
    end function json_raw_value

    logical function json_array_value(text, key, output)
        character(len=*), intent(in) :: text, key
        type(string_t), intent(out) :: output
        integer :: position, after

        call output%clear()
        json_array_value = .false.
        position = json_key(text, key, 1)
        if (position == 0) return
        if (position > len(text)) return
        if (text(position:position) /= '[') return
        if (.not. json_raw_value(text, position, output, after)) return
        json_array_value = output%length() > 1
    end function json_array_value

    integer function json_integer(text, key, fallback)
        character(len=*), intent(in) :: text, key
        integer, intent(in) :: fallback
        integer :: position, last, sign, value
        json_integer = fallback
        position = json_key(text, key, 1)
        if (position == 0) return
        position = skip_space(text, position)
        sign = 1
        if (position <= len(text)) then
            if (text(position:position) == '-') then
                sign = -1
                position = position + 1
            end if
        end if
        last = position
        value = 0
        do while (last <= len(text))
            if (text(last:last) < '0' .or. text(last:last) > '9') exit
            value = min(max_generation, 10 * value + iachar(text(last:last)) - iachar('0'))
            last = last + 1
        end do
        if (last == position) return
        json_integer = max(0, sign * value)
    end function json_integer

    integer(int64) function json_int64(text, key, fallback)
        character(len=*), intent(in) :: text, key
        integer(int64), intent(in) :: fallback
        integer :: position, last, ios
        character(len=:), allocatable :: token

        json_int64 = fallback
        position = json_key(text, key, 1)
        if (position == 0 .or. position > len(text)) return
        last = position
        do while (last <= len(text))
            select case (text(last:last))
            case (',', '}', ']', ' ', char(9), char(10), char(13))
                exit
            case default
                last = last + 1
            end select
        end do
        if (last <= position) return
        token = text(position:last - 1)
        read(token, *, iostat=ios) json_int64
        if (ios /= 0) json_int64 = fallback
    end function json_int64

    real(real32) function json_real(text, key, fallback, valid)
        character(len=*), intent(in) :: text, key
        real(real32), intent(in) :: fallback
        logical, intent(out), optional :: valid
        integer :: position, last, ios
        character(len=:), allocatable :: token

        json_real = fallback
        if (present(valid)) valid = .true.
        position = json_key(text, key, 1)
        if (position == 0) return
        if (position > len(text)) then
            if (present(valid)) valid = .false.
            return
        end if
        last = position
        do while (last <= len(text))
            select case (text(last:last))
            case (',', '}', ']', ' ', char(9), char(10), char(13))
                exit
            case default
                last = last + 1
            end select
        end do
        if (last <= position) then
            if (present(valid)) valid = .false.
            return
        end if
        token = text(position:last - 1)
        read(token, *, iostat=ios) json_real
        if (ios /= 0 .or. .not. finite_real32(json_real)) then
            json_real = fallback
            if (present(valid)) valid = .false.
        end if
    end function json_real

    logical function json_boolean(text, key, fallback)
        character(len=*), intent(in) :: text, key
        logical, intent(in) :: fallback
        integer :: position
        json_boolean = fallback
        position = json_key(text, key, 1)
        if (position == 0) return
        if (position + 3 <= len(text)) then
            if (text(position:position + 3) == 'true') json_boolean = .true.
        end if
        if (position + 4 <= len(text)) then
            if (text(position:position + 4) == 'false') json_boolean = .false.
        end if
    end function json_boolean

    logical function json_object_boolean(text, object_key, key, fallback)
        character(len=*), intent(in) :: text, object_key, key
        logical, intent(in) :: fallback
        integer :: object_value, object_end

        json_object_boolean = fallback
        object_value = json_key(text, object_key, 1)
        if (object_value == 0 .or. object_value > len(text)) return
        if (text(object_value:object_value) /= '{') return
        object_end = matching_delimiter(text, object_value, '{', '}')
        if (object_end == 0) return
        json_object_boolean = json_boolean(text(object_value:object_end), key, fallback)
    end function json_object_boolean

    function json_object_string_value(text, object_key, key, fallback) result(value)
        character(len=*), intent(in) :: text, object_key, key, fallback
        character(len=:), allocatable :: value
        type(string_t) :: parsed
        integer :: object_value, object_end, position, after

        call parsed%set(fallback)
        object_value = json_key(text, object_key, 1)
        if (object_value <= 0) then
            value = parsed%as_character()
            return
        end if
        if (object_value > len(text)) then
            value = parsed%as_character()
            return
        end if
        if (text(object_value:object_value) /= '{') then
            value = parsed%as_character()
            return
        end if
        object_end = matching_delimiter(text, object_value, '{', '}')
        if (object_end == 0) then
            value = parsed%as_character()
            return
        end if
        position = json_key(text, key, object_value, object_end)
        if (position > 0) then
            if (.not. json_string(text, position, parsed, after)) call parsed%set(fallback)
        end if
        value = parsed%as_character()
    end function json_object_string_value

    function json_string_value(text, key, fallback) result(value)
        character(len=*), intent(in) :: text, key, fallback
        character(len=:), allocatable :: value
        type(string_t) :: parsed
        integer :: position, after

        call parsed%set(fallback)
        position = json_key(text, key, 1)
        if (position > 0) then
            if (.not. json_string(text, position, parsed, after)) call parsed%set(fallback)
        end if
        value = parsed%as_character()
    end function json_string_value

    logical function parse_message_array(text, array_key, messages, count)
        character(len=*), intent(in) :: text, array_key
        type(message_t), allocatable, intent(out) :: messages(:)
        integer, intent(out) :: count
        integer :: position, limit, object_end, role_value, content_value, reasoning_value, tool_calls_value, after
        character(len=:), allocatable :: tool_calls_text
        type(message_t), allocatable :: grown(:)
        type(message_t) :: message

        parse_message_array = .false.
        count = 0
        allocate(messages(0))
        position = json_key(text, array_key, 1)
        if (position == 0) return
        if (position > len(text)) return
        if (text(position:position) /= '[') return
        limit = matching_delimiter(text, position, '[', ']')
        if (limit == 0) return
        position = position + 1
        do
            position = skip_space(text, position)
            if (position > limit) exit
            if (text(position:position) == ']') exit
            if (text(position:position) /= '{') return
            if (count >= max_messages) return
            object_end = matching_delimiter(text, position, '{', '}')
            if (object_end == 0) return
            if (object_end > limit) return
            role_value = json_key(text, 'role', position, object_end)
            content_value = json_key(text, 'content', position, object_end)
            reasoning_value = json_key(text, 'reasoning_content', position, object_end)
            tool_calls_value = json_key(text, 'tool_calls', position, object_end)
            if (role_value == 0) return
            if (content_value == 0 .and. reasoning_value == 0 .and. tool_calls_value == 0) return
            if (.not. json_string(text, role_value, message%role, after)) return
            call message%content%clear()
            if (content_value > 0) then
                if (.not. json_content(text, content_value, message%content, after)) return
            end if
            call message%reasoning_content%clear()
            if (reasoning_value > 0) then
                if (.not. json_string(text, reasoning_value, message%reasoning_content, after)) return
            end if
            call message%tool_calls%clear()
            if (tool_calls_value > 0) then
                if (.not. json_raw_value(text, tool_calls_value, message%tool_calls, after)) return
                tool_calls_text = message%tool_calls%as_character()
                if (len(tool_calls_text) > 0) then
                    if (tool_calls_text(1:1) /= '[') return
                end if
            end if
            allocate(grown(count + 1))
            if (count > 0) grown(:count) = messages
            grown(count + 1) = message
            call move_alloc(grown, messages)
            count = count + 1
            position = skip_space(text, object_end + 1)
            if (position > limit) return
            if (text(position:position) == ',') then
                position = position + 1
            else if (text(position:position) == ']') then
                exit
            else
                return
            end if
        end do
        parse_message_array = count > 0
    end function parse_message_array

    logical function parse_messages(text, messages, count)
        character(len=*), intent(in) :: text
        type(message_t), allocatable, intent(out) :: messages(:)
        integer, intent(out) :: count
        integer :: i
        parse_messages = parse_message_array(text, 'messages', messages, count)
        if (.not. parse_messages) return
        do i = 1, count
            if (messages(i)%role%as_character() == 'developer') call messages(i)%role%set('system')
        end do
    end function parse_messages

    logical function parse_response_messages(text, messages, count)
        character(len=*), intent(in) :: text
        type(message_t), allocatable, intent(out) :: messages(:)
        integer, intent(out) :: count
        type(message_t), allocatable :: input_messages(:), grown(:)
        type(message_t) :: instruction
        character(len=:), allocatable :: instructions
        integer :: position, after, input_count, i

        parse_response_messages = .false.
        count = 0
        allocate(messages(0))
        position = json_key(text, 'input', 1)
        if (position == 0) return
        if (position > len(text)) return
        if (text(position:position) == '[') then
            if (.not. parse_message_array(text, 'input', input_messages, input_count)) return
        else if (text(position:position) == '"') then
            allocate(input_messages(1))
            call input_messages(1)%role%set('user')
            if (.not. json_string(text, position, input_messages(1)%content, after)) return
            call input_messages(1)%reasoning_content%clear()
            call input_messages(1)%tool_calls%clear()
            input_count = 1
        else
            return
        end if

        instructions = json_string_value(text, 'instructions', '')
        if (len(instructions) > 0) then
            allocate(grown(input_count + 1))
            call instruction%role%set('system')
            call instruction%content%set(instructions)
            call instruction%reasoning_content%clear()
            call instruction%tool_calls%clear()
            grown(1) = instruction
            if (input_count > 0) grown(2:) = input_messages(:input_count)
            call move_alloc(grown, messages)
            count = input_count + 1
        else
            deallocate(messages)
            allocate(messages(input_count))
            if (input_count > 0) messages = input_messages(:input_count)
            count = input_count
        end if

        do i = 1, count
            if (messages(i)%role%as_character() == 'developer') call messages(i)%role%set('system')
        end do
        parse_response_messages = count > 0
    end function parse_response_messages

    function format_chat(messages, count, enable_thinking, tools_json) result(prompt)
        type(message_t), intent(in) :: messages(:)
        integer, intent(in) :: count
        logical, intent(in) :: enable_thinking
        type(string_t), intent(in) :: tools_json
        type(string_t) :: prompt
        integer :: i, start_index, last_user_index
        character(len=:), allocatable :: role, reasoning, raw_content
        type(string_t) :: content, parsed_reasoning
        logical :: tool_group_open

        call prompt%clear()
        last_user_index = 0
        do i = 1, count
            if (messages(i)%role%as_character() == 'user') then
                raw_content = messages(i)%content%as_character()
                if (.not. is_tool_response(raw_content)) last_user_index = i
            end if
        end do

        start_index = 1
        if (has_tool_entries(tools_json)) then
            if (count > 0) then
                if (messages(1)%role%as_character() == 'system') then
                    call append_tool_system(prompt, tools_json, messages(1)%content, .true.)
                    start_index = 2
                else
                    call content%clear()
                    call append_tool_system(prompt, tools_json, content, .false.)
                end if
            else
                call content%clear()
                call append_tool_system(prompt, tools_json, content, .false.)
            end if
        else if (count > 0) then
            if (messages(1)%role%as_character() == 'system') then
                call prompt%append('<|im_start|>system')
                call prompt%append(char(10))
                call prompt%append_string(messages(1)%content)
                call prompt%append('<|im_end|>')
                call prompt%append(char(10))
                start_index = 2
            end if
        end if

        tool_group_open = .false.
        do i = start_index, count
            role = messages(i)%role%as_character()
            if (role == 'system') cycle
            if (role == 'tool') then
                if (.not. tool_group_open) then
                    call prompt%append('<|im_start|>user')
                    tool_group_open = .true.
                end if
                call prompt%append(char(10))
                call prompt%append('<tool_response>')
                call prompt%append(char(10))
                call prompt%append_string(messages(i)%content)
                call prompt%append(char(10))
                call prompt%append('</tool_response>')
                if (i == count) then
                    call prompt%append('<|im_end|>')
                    call prompt%append(char(10))
                    tool_group_open = .false.
                else
                    role = messages(i + 1)%role%as_character()
                    if (role /= 'tool') then
                        call prompt%append('<|im_end|>')
                        call prompt%append(char(10))
                        tool_group_open = .false.
                    end if
                end if
                cycle
            end if

            call prompt%append('<|im_start|>')
            call prompt%append_string(messages(i)%role)
            call prompt%append(char(10))
            content = messages(i)%content
            raw_content = content%as_character()
            reasoning = messages(i)%reasoning_content%as_character()
            if (role == 'assistant') then
                ! Recover the structured fields when a client sends the
                ! canonical inline Qwen block instead of reasoning_content.
                if (len(reasoning) == 0) then
                    if (index(raw_content, '</think>') > 0) then
                        call split_reasoning(raw_content, .true., 'auto', content, parsed_reasoning)
                        reasoning = parsed_reasoning%as_character()
                    end if
                end if
                ! Qwen3.5 keeps reasoning only for assistant turns after the
                ! latest real user query (including multi-step tool turns).
                if (i > last_user_index) then
                    call prompt%append('<think>')
                    call prompt%append(char(10))
                    call append_trimmed_text(prompt, reasoning)
                    call prompt%append(char(10))
                    call prompt%append('</think>')
                    call prompt%append(char(10))
                    call prompt%append(char(10))
                    call prompt%append_string(content)
                else
                    call prompt%append_string(content)
                end if
                if (messages(i)%tool_calls%length() > 0) then
                    call append_assistant_tool_calls(prompt, messages(i)%tool_calls, content%length() > 0)
                end if
            else
                call prompt%append_string(content)
            end if
            call prompt%append('<|im_end|>')
            call prompt%append(char(10))
        end do
        if (tool_group_open) then
            call prompt%append('<|im_end|>')
            call prompt%append(char(10))
        end if
        call prompt%append('<|im_start|>assistant')
        call prompt%append(char(10))
        if (enable_thinking) then
            call prompt%append('<think>')
            call prompt%append(char(10))
        else
            call prompt%append('<think>')
            call prompt%append(char(10))
            call prompt%append(char(10))
            call prompt%append('</think>')
            call prompt%append(char(10))
            call prompt%append(char(10))
        end if
    end function format_chat

    logical function is_tool_response(text)
        character(len=*), intent(in) :: text
        character(len=*), parameter :: opening = '<tool_response>'
        character(len=*), parameter :: closing = '</tool_response>'
        integer :: first, last

        is_tool_response = .false.
        call trim_bounds(text, first, last)
        if (first > last) return
        if (last - first + 1 < len(opening) + len(closing)) return
        if (text(first:first + len(opening) - 1) /= opening) return
        if (text(last - len(closing) + 1:last) /= closing) return
        is_tool_response = .true.
    end function is_tool_response

    logical function has_tool_entries(value)
        type(string_t), intent(in) :: value
        character(len=:), allocatable :: text

        text = value%as_character()
        has_tool_entries = .false.
        if (len(text) < 3) return
        if (text(1:1) /= '[' .or. text(len(text):len(text)) /= ']') return
        has_tool_entries = index(text, '{') > 0
    end function has_tool_entries

    subroutine trim_bounds(text, first, last)
        character(len=*), intent(in) :: text
        integer, intent(out) :: first, last

        first = 1
        last = len(text)
        do while (first <= last)
            if (iachar(text(first:first)) > 32) exit
            first = first + 1
        end do
        do while (last >= first)
            if (iachar(text(last:last)) > 32) exit
            last = last - 1
        end do
    end subroutine trim_bounds

    subroutine append_trimmed_text(output, text)
        type(string_t), intent(inout) :: output
        character(len=*), intent(in) :: text
        integer :: first, last

        call trim_bounds(text, first, last)
        if (first <= last) call output%append(text(first:last))
    end subroutine append_trimmed_text

    subroutine append_tool_system(prompt, tools_json, system_content, has_system_content)
        type(string_t), intent(inout) :: prompt
        type(string_t), intent(in) :: tools_json, system_content
        logical, intent(in) :: has_system_content

        call prompt%append('<|im_start|>system')
        call prompt%append(char(10))
        call prompt%append('# Tools')
        call prompt%append(char(10))
        call prompt%append(char(10))
        call prompt%append('You have access to the following functions:')
        call prompt%append(char(10))
        call prompt%append(char(10))
        call prompt%append('<tools>')
        call append_tool_array_entries(prompt, tools_json)
        call prompt%append(char(10))
        call prompt%append('</tools>')
        call prompt%append(char(10))
        call prompt%append(char(10))
        call prompt%append('If you choose to call a function ONLY reply in the following format with NO suffix:')
        call prompt%append(char(10))
        call prompt%append(char(10))
        call prompt%append('<tool_call>')
        call prompt%append(char(10))
        call prompt%append('<function=example_function_name>')
        call prompt%append(char(10))
        call prompt%append('<parameter=example_parameter_1>')
        call prompt%append(char(10))
        call prompt%append('value_1')
        call prompt%append(char(10))
        call prompt%append('</parameter>')
        call prompt%append(char(10))
        call prompt%append('<parameter=example_parameter_2>')
        call prompt%append(char(10))
        call prompt%append('This is the value for the second parameter')
        call prompt%append(char(10))
        call prompt%append('that can span')
        call prompt%append(char(10))
        call prompt%append('multiple lines')
        call prompt%append(char(10))
        call prompt%append('</parameter>')
        call prompt%append(char(10))
        call prompt%append('</function>')
        call prompt%append(char(10))
        call prompt%append('</tool_call>')
        call prompt%append(char(10))
        call prompt%append(char(10))
        call prompt%append('<IMPORTANT>')
        call prompt%append(char(10))
        call prompt%append('Reminder:')
        call prompt%append(char(10))
        call prompt%append('- Function calls MUST follow the specified format: an inner <function=...>')
        call prompt%append('</function> block must be nested within <tool_call></tool_call> XML tags')
        call prompt%append(char(10))
        call prompt%append('- Required parameters MUST be specified')
        call prompt%append(char(10))
        call prompt%append('- You may provide optional reasoning for your function call in natural language BEFORE')
        call prompt%append(' the function call, but NOT after')
        call prompt%append(char(10))
        call prompt%append('- If there is no function call available, answer the question like normal with')
        call prompt%append(' your current knowledge and do not tell the user about function calls')
        call prompt%append(char(10))
        call prompt%append('</IMPORTANT>')
        if (has_system_content) then
            call prompt%append(char(10))
            call prompt%append(char(10))
            call append_trimmed_text(prompt, system_content%as_character())
        end if
        call prompt%append('<|im_end|>')
        call prompt%append(char(10))
    end subroutine append_tool_system

    subroutine append_tool_array_entries(prompt, tools_json)
        type(string_t), intent(inout) :: prompt
        type(string_t), intent(in) :: tools_json
        character(len=:), allocatable :: text
        type(string_t) :: item
        integer :: position, limit, after

        text = tools_json%as_character()
        if (len(text) < 2) return
        position = 2
        limit = len(text) - 1
        do
            position = skip_space(text, position)
            if (position > limit) exit
            if (text(position:position) == ']') exit
            if (text(position:position) /= '{') return
            if (.not. json_raw_value(text, position, item, after)) return
            call prompt%append(char(10))
            call prompt%append_string(item)
            position = skip_space(text, after)
            if (position > limit) exit
            if (text(position:position) == ',') then
                position = position + 1
            else if (text(position:position) == ']') then
                exit
            else
                return
            end if
        end do
    end subroutine append_tool_array_entries

    subroutine append_assistant_tool_calls(prompt, raw_calls, has_content)
        type(string_t), intent(inout) :: prompt
        type(string_t), intent(in) :: raw_calls
        logical, intent(in) :: has_content
        character(len=:), allocatable :: text, object_text, function_text
        character(len=:), allocatable :: name
        type(string_t) :: item, arguments
        integer :: position, limit, after, object_end, function_value, function_end, name_value
        integer :: call_index
        logical :: first_call

        text = raw_calls%as_character()
        if (len(text) < 2) return
        if (text(1:1) /= '[') return
        limit = matching_delimiter(text, 1, '[', ']')
        if (limit == 0) return
        position = 2
        call_index = 0
        first_call = .true.
        do
            position = skip_space(text, position)
            if (position >= limit) exit
            if (text(position:position) /= '{') return
            object_end = matching_delimiter(text, position, '{', '}')
            if (object_end == 0 .or. object_end > limit) return
            object_text = text(position:object_end)
            function_value = json_key(object_text, 'function', 1)
            if (function_value > 0) then
                if (function_value > len(object_text)) return
                if (object_text(function_value:function_value) /= '{') return
                function_end = matching_delimiter(object_text, function_value, '{', '}')
                if (function_end == 0) return
                function_text = object_text(function_value:function_end)
            else
                function_text = object_text
            end if
            name_value = json_key(function_text, 'name', 1)
            if (name_value == 0) return
            name = json_string_value(function_text, 'name', '')
            if (len(name) == 0) return
            call arguments%clear()
            name_value = json_key(function_text, 'arguments', 1)
            if (name_value > 0) then
                if (.not. json_raw_value(function_text, name_value, arguments, after)) return
            end if
            call_index = call_index + 1
            if (first_call) then
                if (has_content) then
                    call prompt%append(char(10))
                    call prompt%append(char(10))
                end if
                first_call = .false.
            else
                call prompt%append(char(10))
            end if
            call prompt%append('<tool_call>')
            call prompt%append(char(10))
            call prompt%append('<function=')
            call prompt%append(name)
            call prompt%append('>')
            call prompt%append(char(10))
            call append_json_parameters(prompt, arguments)
            call prompt%append('</function>')
            call prompt%append(char(10))
            call prompt%append('</tool_call>')
            position = skip_space(text, object_end + 1)
            if (position >= limit) exit
            if (text(position:position) == ',') then
                position = position + 1
            else if (text(position:position) == ']') then
                exit
            else
                return
            end if
        end do
    end subroutine append_assistant_tool_calls

    subroutine append_json_parameters(prompt, arguments)
        type(string_t), intent(inout) :: prompt
        type(string_t), intent(in) :: arguments
        character(len=:), allocatable :: text, key, raw_value, value
        type(string_t) :: parsed_key, parsed_value
        integer :: position, limit, colon, after

        text = arguments%as_character()
        if (len(text) < 2) return
        if (text(1:1) /= '{') return
        limit = matching_delimiter(text, 1, '{', '}')
        if (limit == 0) return
        position = 2
        do
            position = skip_space(text, position)
            if (position >= limit) exit
            if (text(position:position) /= '"') return
            if (.not. json_string(text, position, parsed_key, after)) return
            key = parsed_key%as_character()
            colon = skip_space(text, after)
            if (colon > limit) return
            if (text(colon:colon) /= ':') return
            colon = skip_space(text, colon + 1)
            if (.not. json_raw_value(text, colon, parsed_value, after)) return
            raw_value = parsed_value%as_character()
            call prompt%append('<parameter=')
            call prompt%append(key)
            call prompt%append('>')
            call prompt%append(char(10))
            if (len(raw_value) > 0) then
                if (raw_value(1:1) == '"') then
                    if (.not. json_string(raw_value, 1, parsed_value, after)) return
                    value = parsed_value%as_character()
                    call prompt%append(value)
                else
                    call prompt%append(raw_value)
                end if
            end if
            call prompt%append(char(10))
            call prompt%append('</parameter>')
            call prompt%append(char(10))
            position = skip_space(text, after)
            if (position >= limit) exit
            if (text(position:position) == ',') then
                position = position + 1
            else if (text(position:position) == '}') then
                exit
            else
                return
            end if
        end do
    end subroutine append_json_parameters

    subroutine parse_tool_calls(raw, visible, calls, count)
        character(len=*), intent(in) :: raw
        type(string_t), intent(out) :: visible
        type(tool_call_t), allocatable, intent(out) :: calls(:)
        integer, intent(out) :: count
        character(len=*), parameter :: opening = '<tool_call>'
        character(len=*), parameter :: function_opening = '<function='
        character(len=*), parameter :: closing = '</tool_call>'
        character(len=*), parameter :: function_closing = '</function>'
        character(len=:), allocatable :: segment
        type(string_t) :: name, arguments
        integer :: scan, copied, relative, function_relative, open_position, body_position, close_position
        integer :: opening_length, closing_length
        logical :: okay, function_style

        call visible%clear()
        allocate(calls(0))
        count = 0
        if (len(raw) == 0) return
        scan = 1
        copied = 1
        do while (scan <= len(raw))
            relative = index(raw(scan:), opening)
            function_relative = index(raw(scan:), function_opening)
            if (relative == 0) then
                if (function_relative == 0) exit
                function_style = .true.
                relative = function_relative
                opening_length = 0
            else if (function_relative == 0) then
                function_style = .false.
                opening_length = len(opening)
            else if (function_relative < relative) then
                function_style = .true.
                relative = function_relative
                opening_length = 0
            else
                function_style = .false.
                opening_length = len(opening)
            end if
            open_position = scan + relative - 1
            body_position = open_position + opening_length
            if (body_position > len(raw)) exit
            relative = index(raw(body_position:), closing)
            closing_length = len(closing)
            if (relative == 0 .and. function_style) then
                relative = index(raw(body_position:), function_closing)
                closing_length = len(function_closing)
            end if
            if (relative == 0) exit
            close_position = body_position + relative - 1
            if (close_position > body_position) then
                segment = raw(body_position:close_position - 1)
            else
                segment = ''
            end if
            call parse_tool_segment(segment, name, arguments, okay)
            if (okay) then
                if (open_position > copied) call visible%append(raw(copied:open_position - 1))
                call append_tool_call(calls, count, name, arguments)
                copied = close_position + closing_length
                if (function_style) then
                    if (copied + len(closing) - 1 <= len(raw)) then
                        if (raw(copied:copied + len(closing) - 1) == closing) copied = copied + len(closing)
                    end if
                end if
                scan = copied
            else
                scan = close_position + closing_length
            end if
        end do
        if (copied <= len(raw)) call visible%append(raw(copied:))
        if (count == 0) then
            call visible%set(raw)
        else
            call trim_visible_boundaries(visible)
        end if
    end subroutine parse_tool_calls

    subroutine parse_tool_segment(segment, name, arguments, okay)
        character(len=*), intent(in) :: segment
        type(string_t), intent(out) :: name, arguments
        logical, intent(out) :: okay
        character(len=:), allocatable :: function_name, raw_value
        type(string_t) :: parsed_name, parsed_arguments
        integer :: function_position, name_start, name_end, argument_position, after

        call name%clear()
        call arguments%set('{}')
        okay = .false.
        function_position = index(segment, '<function=')
        if (function_position > 0) then
            name_start = function_position + len('<function=')
            if (name_start > len(segment)) return
            name_end = index(segment(name_start:), '>')
            if (name_end == 0) return
            name_end = name_start + name_end - 2
            if (name_end < name_start) return
            function_name = segment(name_start:name_end)
            call name%set(function_name)
            if (name_end + 2 <= len(segment)) then
                call parse_xml_parameters(segment(name_end + 2:), arguments)
            end if
            okay = name%length() > 0
            return
        end if

        call trim_segment(segment, parsed_name)
        raw_value = parsed_name%as_character()
        if (len(raw_value) == 0) return
        if (raw_value(1:1) /= '{') return
        function_name = json_string_value(raw_value, 'name', '')
        if (len(function_name) == 0) return
        call name%set(function_name)
        argument_position = json_key(raw_value, 'arguments', 1)
        if (argument_position > 0) then
            if (.not. json_raw_value(raw_value, argument_position, parsed_arguments, after)) return
            call arguments%set(parsed_arguments%as_character())
        end if
        okay = .true.
    end subroutine parse_tool_segment

    subroutine parse_xml_parameters(text, arguments)
        character(len=*), intent(in) :: text
        type(string_t), intent(out) :: arguments
        character(len=:), allocatable :: parameter_name, parameter_value
        type(string_t) :: key, value
        integer :: scan, relative, parameter_position, name_start, name_end
        integer :: value_start, value_end, close_position, parameter_count

        call arguments%set('{}')
        scan = 1
        parameter_count = 0
        do while (scan <= len(text))
            relative = index(text(scan:), '<parameter=')
            if (relative == 0) exit
            parameter_position = scan + relative - 1
            name_start = parameter_position + len('<parameter=')
            if (name_start > len(text)) exit
            relative = index(text(name_start:), '>')
            if (relative == 0) exit
            name_end = name_start + relative - 2
            if (name_end < name_start) exit
            parameter_name = text(name_start:name_end)
            value_start = name_end + 2
            if (value_start > len(text)) exit
            relative = index(text(value_start:), '</parameter>')
            if (relative == 0) exit
            close_position = value_start + relative - 1
            value_end = close_position - 1
            if (value_end >= value_start) then
                parameter_value = text(value_start:value_end)
            else
                parameter_value = ''
            end if
            call key%set(parameter_name)
            call value%set(parameter_value)
            call append_json_parameter(arguments, parameter_count, key, value)
            scan = close_position + len('</parameter>')
        end do
        if (parameter_count > 0) call arguments%append('}')
    end subroutine parse_xml_parameters

    subroutine append_json_parameter(arguments, parameter_count, key, value)
        type(string_t), intent(inout) :: arguments
        integer, intent(inout) :: parameter_count
        type(string_t), intent(in) :: key, value
        type(string_t) :: escaped_key

        if (parameter_count == 0) then
            call arguments%clear()
            call arguments%append('{')
        else
            call arguments%append(',')
        end if
        escaped_key = json_escape(key)
        call arguments%append_string(escaped_key)
        call arguments%append(':')
        call append_json_argument_value(arguments, value)
        parameter_count = parameter_count + 1
    end subroutine append_json_parameter

    subroutine append_json_argument_value(output, value)
        type(string_t), intent(inout) :: output
        type(string_t), intent(in) :: value
        character(len=:), allocatable :: text
        type(string_t) :: trimmed
        integer :: first, last, delimiter
        logical :: raw_json

        text = value%as_character()
        call trim_bounds(text, first, last)
        if (first > last) then
            call output%append('""')
            return
        end if
        call trimmed%set(text(first:last))
        raw_json = .false.
        if (text(first:first) == '{' .or. text(first:first) == '[') then
            delimiter = matching_delimiter(text, first, text(first:first), &
                merge(']', '}', text(first:first) == '['))
            raw_json = delimiter == last
        else if (text(first:last) == 'true' .or. text(first:last) == 'false' .or. &
            text(first:last) == 'null') then
            raw_json = .true.
        else
            raw_json = numeric_literal(text(first:last))
        end if
        if (raw_json) then
            call output%append(text(first:last))
        else
            call output%append_string(json_escape(trimmed))
        end if
    end subroutine append_json_argument_value

    logical function numeric_literal(text)
        character(len=*), intent(in) :: text
        integer :: i
        logical :: digit_seen

        numeric_literal = .false.
        if (len(text) == 0) return
        digit_seen = .false.
        do i = 1, len(text)
            select case (text(i:i))
            case ('0':'9')
                digit_seen = .true.
            case ('-', '+', '.', 'e', 'E')
                continue
            case default
                return
            end select
        end do
        numeric_literal = digit_seen
    end function numeric_literal

    subroutine append_tool_call(calls, count, name, arguments)
        type(tool_call_t), allocatable, intent(inout) :: calls(:)
        integer, intent(inout) :: count
        type(string_t), intent(in) :: name, arguments
        type(tool_call_t), allocatable :: grown(:)

        allocate(grown(count + 1))
        if (count > 0) grown(:count) = calls
        grown(count + 1)%name = name
        grown(count + 1)%arguments = arguments
        call move_alloc(grown, calls)
        count = count + 1
    end subroutine append_tool_call

    subroutine trim_segment(text, output)
        character(len=*), intent(in) :: text
        type(string_t), intent(out) :: output
        integer :: first, last

        call trim_bounds(text, first, last)
        call output%clear()
        if (first <= last) call output%set(text(first:last))
    end subroutine trim_segment

    subroutine trim_visible_boundaries(value)
        type(string_t), intent(inout) :: value
        character(len=:), allocatable :: text
        integer :: first, last

        text = value%as_character()
        call trim_bounds(text, first, last)
        call value%clear()
        if (first <= last) call value%set(text(first:last))
    end subroutine trim_visible_boundaries

    subroutine split_reasoning(raw, enable_thinking, reasoning_format, content, reasoning)
        character(len=*), intent(in) :: raw, reasoning_format
        logical, intent(in) :: enable_thinking
        type(string_t), intent(out) :: content, reasoning
        integer :: start, finish, close_tag, content_start, prefix_end
        logical :: has_markers

        call content%clear()
        call reasoning%clear()
        if (len(raw) == 0) return
        if (reasoning_format == 'none') then
            call content%set(raw)
            return
        end if
        start = index(raw, '<think>')
        has_markers = start > 0 .or. index(raw, '</think>') > 0
        if (.not. has_markers .and. .not. enable_thinking) then
            call content%set(raw)
            return
        end if
        if (start > 0) then
            prefix_end = start - 1
            start = start + len('<think>')
        else
            prefix_end = 0
            start = 1
        end if
        close_tag = index(raw(start:), '</think>')
        if (close_tag > 0) then
            finish = start + close_tag - 2
            if (finish >= start) call append_trimmed_newlines(reasoning, raw(start:finish))
            content_start = start + close_tag - 1 + len('</think>')
            if (prefix_end > 0) call content%append(raw(:prefix_end))
            if (content_start <= len(raw)) call append_trimmed_newlines(content, raw(content_start:))
        else
            if (prefix_end > 0) call content%append(raw(:prefix_end))
            if (start <= len(raw)) call append_trimmed_newlines(reasoning, raw(start:))
        end if
        if (reasoning_format == 'deepseek-legacy') then
            call content%clear()
            call content%set(raw)
        end if
    end subroutine split_reasoning

    subroutine append_trimmed_newlines(output, text)
        type(string_t), intent(inout) :: output
        character(len=*), intent(in) :: text
        integer :: first, last
        first = 1
        last = len(text)
        do while (first <= last)
            if (text(first:first) /= char(10) .and. text(first:first) /= char(13)) exit
            first = first + 1
        end do
        do while (last >= first)
            if (text(last:last) /= char(10) .and. text(last:last) /= char(13)) exit
            last = last - 1
        end do
        if (first <= last) call output%append(text(first:last))
    end subroutine append_trimmed_newlines

    function json_escape(text) result(escaped)
        type(string_t), intent(in) :: text
        type(string_t) :: escaped
        character(len=:), allocatable :: raw
        integer :: i, value

        raw = text%as_character()
        call escaped%append_char('"')
        do i = 1, len(raw)
            value = iachar(raw(i:i))
            select case (raw(i:i))
            case ('"', '\')
                call escaped%append_char('\'); call escaped%append_char(raw(i:i))
            case (char(10)); call escaped%append('\n')
            case (char(13)); call escaped%append('\r')
            case (char(9)); call escaped%append('\t')
            case default
                if (value < 32) then
                    call escaped%append('\u00')
                    call escaped%append(hex_character(ishft(value, -4)))
                    call escaped%append(hex_character(iand(value, 15)))
                else
                    call escaped%append_char(raw(i:i))
                end if
            end select
        end do
        call escaped%append_char('"')
    end function json_escape

    character(len=1) function hex_character(value)
        integer, intent(in) :: value
        if (value < 10) then
            hex_character = achar(iachar('0') + value)
        else
            hex_character = achar(iachar('a') + value - 10)
        end if
    end function hex_character

    subroutine error_body(status, message, response)
        integer, intent(in) :: status
        character(len=*), intent(in) :: message
        type(string_t), intent(out) :: response
        type(string_t) :: escaped
        type(string_t) :: detail
        call detail%set(message)
        escaped = json_escape(detail)
        call response%clear()
        call response%append('{"error":{"message":')
        call response%append_string(escaped)
        call response%append(',"type":"fortai_server_error"}}')
        call response%append(char(10))
    end subroutine error_body

    function web_ui() result(page)
        type(string_t) :: page
        call page%append('<!doctype html><html lang="en"><head><meta charset="utf-8">')
        call page%append('<meta name="viewport" content="width=device-width,initial-scale=1">')
        call page%append('<title>FortAI</title><style>')
        call page%append(':root{color-scheme:dark;font:16px system-ui,sans-serif}body{margin:0;background:#10131a;color:#edf2f7}')
        call page%append('main{max-width:960px;margin:auto;padding:28px}.brand{display:flex;align-items:center;gap:12px}')
        call page%append('.dot{width:13px;height:13px;border-radius:50%;background:#53e6a7;box-shadow:0 0 18px #53e6a7}')
        call page%append('h1{font-size:1.5rem;margin:0}.sub{color:#9ba7b5;margin:4px 0 24px}.chat{min-height:55vh;border:1px solid #2d3748;border-radius:14px;padding:18px;background:#151a23;overflow:auto}')
        call page%append('.msg{white-space:pre-wrap;line-height:1.5;margin:12px 0;padding:12px 15px;border-radius:10px;max-width:85%}.user{margin-left:auto;background:#1d4d43}.assistant{background:#222b39}.role{font-size:.75rem;color:#9ba7b5;text-transform:uppercase;letter-spacing:.08em;margin-bottom:5px}.thought{margin-bottom:10px;color:#aeb8c5;font-size:.9rem}.thought summary{cursor:pointer;color:#73d6b1;margin-bottom:6px}')
        call page%append('form{display:flex;gap:10px;margin-top:14px}textarea{flex:1;resize:vertical;min-height:58px;max-height:240px;border:1px solid #39475a;border-radius:10px;padding:12px;background:#0e1117;color:inherit;font:inherit}button{border:0;border-radius:10px;padding:0 22px;background:#53e6a7;color:#0b2118;font-weight:700;cursor:pointer}.status{color:#9ba7b5;font-size:.85rem;margin-top:9px}</style></head><body><main>')
        call page%append('<div class=brand><span class=dot></span><div><h1>FortAI</h1><div class=sub>Native Qwen3.5 CUDA inference</div></div></div>')
        call page%append('<section id=chat class="chat"><div class="msg assistant"><div class=role>FortAI</div>Ready.</div></section>')
        call page%append('<form id=form><textarea id=input placeholder="Message FortAI…"></textarea><button id=send>Send</button></form><div id=status class=status>Connected to the FortAI-native server.</div></main><script>')
        call page%append('const chat=document.querySelector("#chat"),input=document.querySelector("#input"),form=document.querySelector("#form"),send=document.querySelector("#send"),status=document.querySelector("#status"),messages=[];')
        call page%append('function add(role,text){const el=document.createElement("div");el.className="msg "+role;const r=document.createElement("div");r.className="role";r.textContent=role==="user"?"You":"FortAI";const t=document.createElement("div");t.textContent=text;el.append(r,t);chat.append(el);chat.scrollTop=chat.scrollHeight;return t}')
        call page%append('form.addEventListener("submit",async e=>{e.preventDefault();const text=input.value.trim();if(!text)return;input.value="";add("user",text);messages.push({role:"user",content:text});send.disabled=true;status.textContent="Generating…";const target=add("assistant","");try{const r=await fetch("/v1/chat/completions",{method:"POST",headers:{"content-type":"application/json"},body:JSON.stringify({model:"qwen",messages,max_tokens:512,stream:false,temperature:0})});const d=await r.json();if(!r.ok)throw Error(d.error?.message||"request failed");const m=d.choices?.[0]?.message||{};target.textContent=m.content||"";if(m.reasoning_content){const details=document.createElement("details"),summary=document.createElement("summary"),thought=document.createElement("div");details.className="thought";summary.textContent="Thinking";thought.textContent=m.reasoning_content;details.append(summary,thought);target.parentElement.insertBefore(details,target)}messages.push({role:"assistant",content:m.content||"",reasoning_content:m.reasoning_content||""});status.textContent="FortAI native CUDA"}catch(err){target.textContent="Error: "+err.message;status.textContent="Request failed"}finally{send.disabled=false;input.focus()}})</script></body></html>')
    end function web_ui

    subroutine parse_http_request(request, method, path, body, okay)
        type(string_t), intent(in) :: request
        type(string_t), intent(out) :: method, path, body
        logical, intent(out) :: okay
        character(len=:), allocatable :: raw, line
        integer :: first_space, second_space, header_end, query

        call method%clear(); call path%clear(); call body%clear(); okay = .false.
        raw = request%as_character()
        first_space = index(raw, ' ')
        if (first_space <= 1) return
        second_space = index(raw(first_space + 1:), ' ')
        if (second_space <= 1) return
        second_space = second_space + first_space
        header_end = index(raw, char(13) // char(10) // char(13) // char(10))
        if (header_end == 0) return
        line = raw(first_space + 1:second_space - 1)
        query = index(line, '?')
        if (query > 0) line = line(:query - 1)
        call method%set(raw(:first_space - 1))
        call path%set(line)
        if (header_end + 4 <= len(raw)) call body%set(raw(header_end + 4:))
        okay = .true.
    end subroutine parse_http_request

    integer function unix_timestamp()
        integer, parameter :: days_in_month(12) = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        integer :: calendar(8), year, month, day, leap_days

        call date_and_time(values=calendar)
        year = calendar(1)
        month = calendar(2)
        day = calendar(3)
        unix_timestamp = 0
        do while (year > 1970)
            year = year - 1
            leap_days = merge(1, 0, mod(year, 4) == 0 .and. (mod(year, 100) /= 0 .or. mod(year, 400) == 0))
            unix_timestamp = unix_timestamp + 365 + leap_days
        end do
        do month = 1, calendar(2) - 1
            unix_timestamp = unix_timestamp + days_in_month(month)
            if (month == 2 .and. mod(calendar(1), 4) == 0 .and. &
                (mod(calendar(1), 100) /= 0 .or. mod(calendar(1), 400) == 0)) unix_timestamp = unix_timestamp + 1
        end do
        unix_timestamp = 86400 * (unix_timestamp + day - 1) + 3600 * calendar(5) + &
            60 * calendar(6) + calendar(7)
    end function unix_timestamp

    subroutine completion_body(model, content, reasoning, tokens, chat, stream, response, tool_calls, tool_count)
        type(string_t), intent(in) :: model, content, reasoning
        type(tool_call_t), intent(in) :: tool_calls(:)
        integer, intent(in) :: tokens, tool_count
        logical, intent(in) :: chat, stream
        type(string_t), intent(out) :: response
        type(string_t) :: model_json, text_json, reasoning_json, escaped
        integer :: clock, i

        call response%clear()
        clock = unix_timestamp()
        model_json = json_escape(model)
        text_json = json_escape(content)
        if (reasoning%length() > 0) reasoning_json = json_escape(reasoning)
        if (chat .and. stream) then
            ! Reasoning, visible content, and function calls are independent
            ! deltas so OpenAI clients can render or dispatch each channel.
            if (reasoning%length() > 0) then
                call response%append('data: {"id":"fortai-native","object":"chat.completion.chunk","created":')
                call response%append_int(clock)
                call response%append(',"model":')
                call response%append_string(model_json)
                call response%append(',"choices":[{"index":0,"delta":{"role":"assistant","reasoning_content":')
                call response%append_string(reasoning_json)
                call response%append('},"finish_reason":null}],"fortai_backend":"fortai"}')
                call response%append(char(10)); call response%append(char(10))
            end if
            if (content%length() > 0 .or. tool_count == 0) then
                call response%append('data: {"id":"fortai-native","object":"chat.completion.chunk","created":')
                call response%append_int(clock)
                call response%append(',"model":')
                call response%append_string(model_json)
                if (reasoning%length() == 0) then
                    call response%append(',"choices":[{"index":0,"delta":{"role":"assistant","content":')
                else
                    call response%append(',"choices":[{"index":0,"delta":{"content":')
                end if
                call response%append_string(text_json)
                call response%append('},"finish_reason":null}],"fortai_backend":"fortai"}')
                call response%append(char(10)); call response%append(char(10))
            end if
            do i = 1, tool_count
                escaped = json_escape(tool_calls(i)%arguments)
                call response%append('data: {"id":"fortai-native","object":"chat.completion.chunk","created":')
                call response%append_int(clock)
                call response%append(',"model":')
                call response%append_string(model_json)
                call response%append(',"choices":[{"index":0,"delta":{"tool_calls":[{"index":')
                call response%append_int(i - 1)
                call response%append(',"id":"fortai-tool-call-')
                call response%append_int(i)
                call response%append('","type":"function","function":{"name":')
                call response%append_string(json_escape(tool_calls(i)%name))
                call response%append(',"arguments":')
                call response%append_string(escaped)
                call response%append('}}]},"finish_reason":null}],"fortai_backend":"fortai"}')
                call response%append(char(10)); call response%append(char(10))
            end do
            call response%append('data: {"id":"fortai-native","object":"chat.completion.chunk","created":')
            call response%append_int(clock)
            call response%append(',"model":')
            call response%append_string(model_json)
            call response%append(',"choices":[{"index":0,"delta":{},"finish_reason":')
            if (tool_count > 0) then
                call response%append('"tool_calls"')
            else
                call response%append('"stop"')
            end if
            call response%append('}],"fortai_backend":"fortai"}')
            call response%append(char(10)); call response%append(char(10))
            call response%append('data: [DONE]'); call response%append(char(10)); call response%append(char(10))
            return
        end if
        call response%append('{"id":"fortai-native","object":')
        if (chat) then
            call response%append('"chat.completion"')
        else
            call response%append('"text_completion"')
        end if
        call response%append(',"created":')
        call response%append_int(clock)
        call response%append(',"model":')
        call response%append_string(model_json)
        if (chat) then
            call response%append(',"choices":[{"index":0,"message":{"role":"assistant","content":')
            if (content%length() == 0 .and. tool_count > 0) then
                call response%append('null')
            else
                call response%append_string(text_json)
            end if
            if (reasoning%length() > 0) then
                call response%append(',"reasoning_content":')
                call response%append_string(reasoning_json)
            end if
            if (tool_count > 0) then
                call response%append(',"tool_calls":[')
                do i = 1, tool_count
                    if (i > 1) call response%append(',')
                    call append_chat_tool_object(response, tool_calls(i), i)
                end do
                call response%append(']')
            end if
            call response%append('},"finish_reason":')
            if (tool_count > 0) then
                call response%append('"tool_calls"')
            else
                call response%append('"stop"')
            end if
            call response%append('}],"usage":{"prompt_tokens":0,"completion_tokens":')
            call response%append_int(tokens); call response%append(',"total_tokens":'); call response%append_int(tokens)
            call response%append('},"fortai_backend":"fortai"}'); call response%append(char(10))
        else
            call response%append(',"choices":[{"index":0,"text":')
            call response%append_string(text_json)
            call response%append(',"finish_reason":"stop"}],"usage":{"prompt_tokens":0,"completion_tokens":')
            call response%append_int(tokens); call response%append(',"total_tokens":'); call response%append_int(tokens)
            call response%append('},"fortai_backend":"fortai"}'); call response%append(char(10))
        end if
    end subroutine completion_body

    subroutine append_chat_tool_object(output, call, index)
        type(string_t), intent(inout) :: output
        type(tool_call_t), intent(in) :: call
        integer, intent(in) :: index
        type(string_t) :: arguments

        arguments = json_escape(call%arguments)
        call output%append('{"id":"fortai-tool-call-')
        call output%append_int(index)
        call output%append('","type":"function","function":{"name":')
        call output%append_string(json_escape(call%name))
        call output%append(',"arguments":')
        call output%append_string(arguments)
        call output%append('}}')
    end subroutine append_chat_tool_object

    function response_json(model, content, reasoning, tokens, tool_calls, tool_count) result(response)
        type(string_t), intent(in) :: model, content, reasoning
        type(tool_call_t), intent(in) :: tool_calls(:)
        integer, intent(in) :: tokens, tool_count
        type(string_t) :: response
        type(string_t) :: model_json, content_json, reasoning_json, arguments
        integer :: clock, i
        logical :: have_output

        call response%clear()
        clock = unix_timestamp()
        model_json = json_escape(model)
        content_json = json_escape(content)
        if (reasoning%length() > 0) reasoning_json = json_escape(reasoning)
        call response%append('{"id":"fortai-native-response","object":"response","created_at":')
        call response%append_int(clock)
        call response%append(',"completed_at":')
        call response%append_int(clock)
        call response%append(',"status":"completed","model":')
        call response%append_string(model_json)
        call response%append(',"output":[')
        have_output = .false.
        if (reasoning%length() > 0) then
            call response%append('{"id":"fortai-native-reasoning","type":"reasoning","status":"completed",')
            call response%append('"summary":[{"type":"summary_text","text":')
            call response%append_string(reasoning_json)
            call response%append('}]}')
            have_output = .true.
        end if
        if (content%length() > 0 .or. tool_count == 0) then
            if (have_output) call response%append(',')
            call response%append('{"id":"fortai-native-message","type":"message","role":"assistant",')
            call response%append('"status":"completed","content":[{"type":"output_text","text":')
            call response%append_string(content_json)
            call response%append(',"annotations":[]}]}')
            have_output = .true.
        end if
        do i = 1, tool_count
            if (have_output) call response%append(',')
            arguments = json_escape(tool_calls(i)%arguments)
            call response%append('{"id":"fortai-tool-call-')
            call response%append_int(i)
            call response%append('","type":"function_call","status":"completed",')
            call response%append('"call_id":"fortai-tool-call-')
            call response%append_int(i)
            call response%append('","name":')
            call response%append_string(json_escape(tool_calls(i)%name))
            call response%append(',"arguments":')
            call response%append_string(arguments)
            call response%append('}')
            have_output = .true.
        end do
        call response%append('],"input":[],"instructions":null,"usage":{"input_tokens":0,"output_tokens":')
        call response%append_int(tokens)
        call response%append(',"total_tokens":')
        call response%append_int(tokens)
        call response%append('},"store":false,"fortai_backend":"fortai"}')
    end function response_json

    subroutine append_sse_event(response, event_name, payload)
        type(string_t), intent(inout) :: response
        character(len=*), intent(in) :: event_name
        type(string_t), intent(in) :: payload
        call response%append('event: ')
        call response%append(trim(event_name))
        call response%append(char(10))
        call response%append('data: ')
        call response%append_string(payload)
        call response%append(char(10))
        call response%append(char(10))
    end subroutine append_sse_event

    function response_stream_json(model, content, reasoning, tokens, tool_calls, tool_count) result(response)
        type(string_t), intent(in) :: model, content, reasoning
        type(tool_call_t), intent(in) :: tool_calls(:)
        integer, intent(in) :: tokens, tool_count
        type(string_t) :: response, payload, escaped, final_response, arguments
        integer :: text_index, clock, i, tool_index
        logical :: has_message

        call response%clear()
        clock = unix_timestamp()
        call payload%set('{"type":"response.created","response":{"id":"fortai-native-response",')
        call payload%append('"object":"response","created_at":')
        call payload%append_int(clock)
        call payload%append(',"status":"in_progress","completed_at":null,"model":')
        call payload%append_string(json_escape(model))
        call payload%append(',"output":[],"usage":null}}')
        call append_sse_event(response, 'response.created', payload)
        call payload%set('{"type":"response.in_progress","response":{"id":"fortai-native-response",')
        call payload%append('"object":"response","created_at":')
        call payload%append_int(clock)
        call payload%append(',"status":"in_progress","completed_at":null,"model":')
        call payload%append_string(json_escape(model))
        call payload%append(',"output":[],"usage":null}}')
        call append_sse_event(response, 'response.in_progress', payload)

        text_index = 0
        if (reasoning%length() > 0) then
            call payload%set('{"type":"response.output_item.added","output_index":0,"item":')
            call payload%append('{"id":"fortai-native-reasoning","type":"reasoning","status":"in_progress",')
            call payload%append('"summary":[]}}')
            call append_sse_event(response, 'response.output_item.added', payload)
            escaped = json_escape(reasoning)
            call payload%set('{"type":"response.reasoning_summary_text.delta","item_id":"fortai-native-reasoning",')
            call payload%append('"output_index":0,"summary_index":0,"delta":')
            call payload%append_string(escaped)
            call payload%append('}')
            call append_sse_event(response, 'response.reasoning_summary_text.delta', payload)
            call payload%set('{"type":"response.reasoning_summary_part.done","item_id":"fortai-native-reasoning",')
            call payload%append('"output_index":0,"summary_index":0,"part":{"type":"summary_text","text":')
            call payload%append_string(escaped)
            call payload%append('}}')
            call append_sse_event(response, 'response.reasoning_summary_part.done', payload)
            call payload%set('{"type":"response.output_item.done","output_index":0,"item":')
            call payload%append('{"id":"fortai-native-reasoning","type":"reasoning","status":"completed",')
            call payload%append('"summary":[{"type":"summary_text","text":')
            call payload%append_string(escaped)
            call payload%append('}]}}')
            call append_sse_event(response, 'response.output_item.done', payload)
            text_index = 1
        end if

        has_message = content%length() > 0 .or. tool_count == 0
        if (has_message) then
        call payload%set('{"type":"response.output_item.added","output_index":')
        call payload%append_int(text_index)
        call payload%append(',"item":{"id":"fortai-native-message","type":"message",')
        call payload%append('"role":"assistant","status":"in_progress","content":[]}}')
        call append_sse_event(response, 'response.output_item.added', payload)
        call payload%set('{"type":"response.content_part.added","item_id":"fortai-native-message",')
        call payload%append('"output_index":')
        call payload%append_int(text_index)
        call payload%append(',"content_index":0,"part":{"type":"output_text","text":"","annotations":[]}}')
        call append_sse_event(response, 'response.content_part.added', payload)
        escaped = json_escape(content)
        call payload%set('{"type":"response.output_text.delta","item_id":"fortai-native-message",')
        call payload%append('"output_index":')
        call payload%append_int(text_index)
        call payload%append(',"content_index":0,"delta":')
        call payload%append_string(escaped)
        call payload%append('}')
        call append_sse_event(response, 'response.output_text.delta', payload)
        call payload%set('{"type":"response.output_text.done","item_id":"fortai-native-message",')
        call payload%append('"output_index":')
        call payload%append_int(text_index)
        call payload%append(',"content_index":0,"text":')
        call payload%append_string(escaped)
        call payload%append('}')
        call append_sse_event(response, 'response.output_text.done', payload)
        call payload%set('{"type":"response.content_part.done","item_id":"fortai-native-message",')
        call payload%append('"output_index":')
        call payload%append_int(text_index)
        call payload%append(',"content_index":0,"part":{"type":"output_text","text":')
        call payload%append_string(escaped)
        call payload%append(',"annotations":[]}}')
        call append_sse_event(response, 'response.content_part.done', payload)
        call payload%set('{"type":"response.output_item.done","output_index":')
        call payload%append_int(text_index)
        call payload%append(',"item":{"id":"fortai-native-message","type":"message",')
        call payload%append('"role":"assistant","status":"completed","content":[{"type":"output_text",')
        call payload%append('"text":')
        call payload%append_string(escaped)
        call payload%append(',"annotations":[]}]}}')
        call append_sse_event(response, 'response.output_item.done', payload)
        end if

        tool_index = text_index
        if (has_message) tool_index = tool_index + 1
        do i = 1, tool_count
            arguments = json_escape(tool_calls(i)%arguments)
            call payload%set('{"type":"response.output_item.added","output_index":')
            call payload%append_int(tool_index)
            call payload%append(',"item":{"id":"fortai-tool-call-')
            call payload%append_int(i)
            call payload%append('","type":"function_call","status":"in_progress",')
            call payload%append('"call_id":"fortai-tool-call-')
            call payload%append_int(i)
            call payload%append('","name":')
            call payload%append_string(json_escape(tool_calls(i)%name))
            call payload%append(',"arguments":""}}')
            call append_sse_event(response, 'response.output_item.added', payload)
            call payload%set('{"type":"response.function_call_arguments.delta","item_id":"fortai-tool-call-')
            call payload%append_int(i)
            call payload%append('","output_index":')
            call payload%append_int(tool_index)
            call payload%append(',"delta":')
            call payload%append_string(arguments)
            call payload%append('}')
            call append_sse_event(response, 'response.function_call_arguments.delta', payload)
            call payload%set('{"type":"response.function_call_arguments.done","item_id":"fortai-tool-call-')
            call payload%append_int(i)
            call payload%append('","output_index":')
            call payload%append_int(tool_index)
            call payload%append(',"arguments":')
            call payload%append_string(arguments)
            call payload%append('}')
            call append_sse_event(response, 'response.function_call_arguments.done', payload)
            call payload%set('{"type":"response.output_item.done","output_index":')
            call payload%append_int(tool_index)
            call payload%append(',"item":{"id":"fortai-tool-call-')
            call payload%append_int(i)
            call payload%append('","type":"function_call","status":"completed",')
            call payload%append('"call_id":"fortai-tool-call-')
            call payload%append_int(i)
            call payload%append('","name":')
            call payload%append_string(json_escape(tool_calls(i)%name))
            call payload%append(',"arguments":')
            call payload%append_string(arguments)
            call payload%append('}}')
            call append_sse_event(response, 'response.output_item.done', payload)
            tool_index = tool_index + 1
        end do

        final_response = response_json(model, content, reasoning, tokens, tool_calls, tool_count)
        call payload%set('{"type":"response.completed","response":')
        call payload%append_string(final_response)
        call payload%append('}')
        call append_sse_event(response, 'response.completed', payload)
    end function response_stream_json

    integer(c_int) function fortai_native_http_handle(request, request_length, model, cuda, response, &
            response_capacity, response_length, status, content_type, content_type_capacity) &
            bind(C, name='fortai_native_http_handle')
        character(kind=c_char), intent(in) :: request(*), model(*)
        integer(c_int), value, intent(in) :: request_length, cuda, response_capacity, content_type_capacity
        character(kind=c_char), intent(out) :: response(*), content_type(*)
        integer(c_int), intent(out) :: response_length, status
        type(string_t) :: request_text, model_text, method, path, body, result, prompt, generated
        type(string_t) :: content, reasoning, tools_json, clean_content
        type(message_t), allocatable :: messages(:)
        type(tool_call_t), allocatable :: tool_calls(:)
        integer :: count, max_tokens, tokens, result_code, required, tool_count
        integer(int64) :: seed
        real(real32) :: temperature
        logical :: okay, chat, responses, stream, enable_thinking, temperature_valid
        character(len=:), allocatable :: path_value
        character(len=:), allocatable :: reasoning_format, reasoning_effort

        response_length = 0_c_int
        status = 500_c_int
        call request_text%from_c(request, int(request_length))
        call model_text%from_c(model)
        call parse_http_request(request_text, method, path, body, okay)
        if (.not. okay) then
            call error_body(400, 'malformed HTTP request', result); status = 400_c_int
            call copy_result(result, 'application/json', response, response_capacity, response_length, &
                content_type, content_type_capacity, fortai_native_http_handle); return
        end if
        path_value = path%as_character()
        if (path_value == '/' .or. path_value == '/ui' .or. path_value == '/index.html') then
            result = web_ui(); status = 200_c_int
            call copy_result(result, 'text/html; charset=utf-8', response, response_capacity, response_length, &
                content_type, content_type_capacity, fortai_native_http_handle); return
        end if
        if (path_value == '/health' .or. path_value == '/v1/health') then
            call result%append('{"status":"ok","backend":"fortai","engine":"fortai-native-qwen35","cuda":')
            call result%append_logical(cuda /= 0_c_int)
            call result%append(',"service":"fortai-server"}'); call result%append(char(10)); status = 200_c_int
            call copy_result(result, 'application/json', response, response_capacity, response_length, &
                content_type, content_type_capacity, fortai_native_http_handle); return
        end if
        if (path_value == '/v1/models' .and. method%as_character() /= 'POST') then
            call result%append('{"object":"list","data":[{"id":'); result = append_json(result, model_text)
            call result%append(',"object":"model","owned_by":"fortai","fortai_backend":"fortai"}]}'); call result%append(char(10))
            status = 200_c_int
            call copy_result(result, 'application/json', response, response_capacity, response_length, &
                content_type, content_type_capacity, fortai_native_http_handle); return
        end if
        responses = path_value == '/v1/responses'
        if (method%as_character() /= 'POST' .or. ((path_value /= '/v1/chat/completions') .and. &
            (path_value /= '/chat/completions') .and. (path_value /= '/v1/completions') .and. &
            (path_value /= '/completion') .and. .not. responses)) then
            call error_body(404, 'endpoint not available', result); status = 404_c_int
            call copy_result(result, 'application/json', response, response_capacity, response_length, &
                content_type, content_type_capacity, fortai_native_http_handle); return
        end if
        chat = index(path_value, 'chat') > 0
        if (responses) then
            max_tokens = json_integer(body%as_character(), 'max_output_tokens', &
                json_integer(body%as_character(), 'max_completion_tokens', &
                json_integer(body%as_character(), 'max_tokens', 128)))
        else
            max_tokens = json_integer(body%as_character(), 'max_tokens', &
                json_integer(body%as_character(), 'max_completion_tokens', 128))
        end if
        temperature = json_real(body%as_character(), 'temperature', 0.0_real32, temperature_valid)
        if (.not. temperature_valid .or. .not. finite_real32(temperature) .or. temperature < 0.0_real32) then
            call error_body(400, 'temperature must be a finite non-negative number', result); status = 400_c_int
            call copy_result(result, 'application/json', response, response_capacity, response_length, &
                content_type, content_type_capacity, fortai_native_http_handle); return
        end if
        seed = json_int64(body%as_character(), 'seed', 0_int64)
        stream = json_boolean(body%as_character(), 'stream', .false.)
        enable_thinking = json_boolean(body%as_character(), 'enable_thinking', .true.)
        ! OpenAI-compatible clients commonly pass this through
        ! chat_template_kwargs, which has precedence in llama.cpp.
        enable_thinking = json_object_boolean(body%as_character(), 'chat_template_kwargs', &
            'enable_thinking', enable_thinking)
        reasoning_effort = json_object_string_value(body%as_character(), 'reasoning', 'effort', '')
        select case (trim(reasoning_effort))
        case ('none', 'off')
            enable_thinking = .false.
        case ('minimal', 'low', 'medium', 'high', 'xhigh')
            enable_thinking = .true.
        end select
        reasoning_format = json_string_value(body%as_character(), 'reasoning_format', 'auto')
        call tools_json%clear()
        if (.not. json_array_value(body%as_character(), 'tools', tools_json)) call tools_json%clear()
        if (responses) then
            if (.not. parse_response_messages(body%as_character(), messages, count)) then
                call error_body(400, 'input must contain a string or message array', result); status = 400_c_int
                call copy_result(result, 'application/json', response, response_capacity, response_length, &
                    content_type, content_type_capacity, fortai_native_http_handle); return
            end if
            prompt = format_chat(messages, count, enable_thinking, tools_json)
        else if (chat) then
            if (.not. parse_messages(body%as_character(), messages, count)) then
                call error_body(400, 'messages must contain role/content strings', result); status = 400_c_int
                call copy_result(result, 'application/json', response, response_capacity, response_length, &
                    content_type, content_type_capacity, fortai_native_http_handle); return
            end if
            prompt = format_chat(messages, count, enable_thinking, tools_json)
        else
            block
            integer :: prompt_value, after
            prompt_value = json_key(body%as_character(), 'prompt', 1)
            if (prompt_value == 0) prompt_value = json_key(body%as_character(), 'content', 1)
            if (prompt_value == 0 .or. .not. json_string(body%as_character(), prompt_value, prompt, after)) then
                call error_body(400, 'prompt must be a string', result); status = 400_c_int
                call copy_result(result, 'application/json', response, response_capacity, response_length, &
                    content_type, content_type_capacity, fortai_native_http_handle); return
            end if
            end block
        end if
        result_code = fortai_native_service_complete_text_options(prompt%as_character(), max_tokens, &
            temperature, seed, generated, tokens)
        if (result_code < 0) then
            call error_body(500, 'FortAI generation failed', result); status = 500_c_int
            call copy_result(result, 'application/json', response, response_capacity, response_length, &
                content_type, content_type_capacity, fortai_native_http_handle); return
        end if
        if (responses .or. chat) then
            call split_reasoning(generated%as_character(), enable_thinking, reasoning_format, content, reasoning)
            call parse_tool_calls(content%as_character(), clean_content, tool_calls, tool_count)
            content = clean_content
        else
            content = generated
            call reasoning%clear()
            allocate(tool_calls(0))
            tool_count = 0
        end if
        if (responses) then
            if (stream) then
                result = response_stream_json(model_text, content, reasoning, tokens, tool_calls, tool_count)
            else
                result = response_json(model_text, content, reasoning, tokens, tool_calls, tool_count)
            end if
        else
            call completion_body(model_text, content, reasoning, tokens, chat, stream, result, tool_calls, tool_count)
        end if
        status = 200_c_int
        block
            character(len=32) :: mime
            mime = 'application/json'
            if ((chat .or. responses) .and. stream) mime = 'text/event-stream'
            call copy_result(result, mime(:len_trim(mime)), response, response_capacity, response_length, &
                content_type, content_type_capacity, fortai_native_http_handle)
        end block
    end function fortai_native_http_handle

    function append_json(base, value) result(output)
        type(string_t), intent(in) :: base, value
        type(string_t) :: output, escaped
        output = base
        escaped = json_escape(value)
        call output%append_string(escaped)
    end function append_json

    subroutine copy_result(result, mime, response, capacity, response_length, content_type, &
            content_type_capacity, code)
        type(string_t), intent(in) :: result
        character(len=*), intent(in) :: mime
        character(kind=c_char), intent(out) :: response(*), content_type(*)
        integer(c_int), intent(in) :: capacity, content_type_capacity
        integer(c_int), intent(out) :: response_length
        integer(c_int), intent(inout) :: code
        integer :: required
        response_length = int(result%length(), c_int)
        if (result%length() >= int(capacity) .or. len(mime) + 1 >= int(content_type_capacity)) then
            code = -int(max(result%length() + 1, len(mime) + 1), c_int)
            return
        end if
        call result%to_c(response, int(capacity), required)
        call string_to_c_literal(mime, content_type, int(content_type_capacity))
        code = 0_c_int
    end subroutine copy_result

    subroutine string_to_c_literal(text, output, capacity)
        character(len=*), intent(in) :: text
        character(kind=c_char), intent(out) :: output(*)
        integer, intent(in) :: capacity
        integer :: i
        do i = 1, min(len(text), capacity - 1)
            output(i) = text(i:i)
        end do
        output(min(len(text), capacity - 1) + 1) = c_null_char
    end subroutine string_to_c_literal

end module fortai_native_http
