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
    end type message_t

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
        integer :: limit, value

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
        value = json_key(text, 'text', first, limit)
        if (value == 0) then
            json_content = .false.
            return
        end if
        if (text(value:value) /= '"') then
            json_content = .false.
            return
        end if
        json_content = json_string(text, value, output, after)
    end function json_content

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

    logical function parse_messages(text, messages, count)
        character(len=*), intent(in) :: text
        type(message_t), allocatable, intent(out) :: messages(:)
        integer, intent(out) :: count
        integer :: position, limit, object_end, role_value, content_value, reasoning_value, after
        type(message_t), allocatable :: grown(:)
        type(message_t) :: message

        parse_messages = .false.
        count = 0
        allocate(messages(0))
        position = json_key(text, 'messages', 1)
        if (position == 0) return
        if (text(position:position) /= '[') return
        limit = matching_delimiter(text, position, '[', ']')
        if (limit == 0) return
        position = position + 1
        do
            position = skip_space(text, position)
            if (position > limit) exit
            if (text(position:position) == ']') exit
            if (text(position:position) /= '{' .or. count >= max_messages) return
            object_end = matching_delimiter(text, position, '{', '}')
            if (object_end == 0 .or. object_end > limit) return
            role_value = json_key(text, 'role', position, object_end)
            content_value = json_key(text, 'content', position, object_end)
            reasoning_value = json_key(text, 'reasoning_content', position, object_end)
            if (role_value == 0 .or. (content_value == 0 .and. reasoning_value == 0)) return
            if (.not. json_string(text, role_value, message%role, after)) return
            call message%content%clear()
            if (content_value > 0) then
                if (.not. json_content(text, content_value, message%content, after)) return
            end if
            call message%reasoning_content%clear()
            if (reasoning_value > 0) then
                if (.not. json_string(text, reasoning_value, message%reasoning_content, after)) return
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
        parse_messages = count > 0
    end function parse_messages

    function format_chat(messages, count, enable_thinking) result(prompt)
        type(message_t), intent(in) :: messages(:)
        integer, intent(in) :: count
        logical, intent(in) :: enable_thinking
        type(string_t) :: prompt
        integer :: i, last_user_index
        character(len=:), allocatable :: role, reasoning, raw_content
        type(string_t) :: content, parsed_reasoning
        call prompt%clear()
        last_user_index = 0
        do i = 1, count
            if (messages(i)%role%as_character() == 'user') last_user_index = i
        end do
        do i = 1, count
            call prompt%append('<|im_start|>')
            call prompt%append_string(messages(i)%role)
            call prompt%append(char(10))
            role = messages(i)%role%as_character()
            reasoning = messages(i)%reasoning_content%as_character()
            content = messages(i)%content
            raw_content = content%as_character()
            ! Qwen's template closes every historical assistant thinking block,
            ! even when it is empty.  If a client sent the legacy inline form,
            ! recover its two fields before rebuilding the canonical block.
            if (role == 'assistant') then
                if (len(reasoning) == 0 .and. index(raw_content, '</think>') > 0) then
                    call split_reasoning(raw_content, .true., 'auto', content, parsed_reasoning)
                    reasoning = parsed_reasoning%as_character()
                end if
                ! Qwen3.5 only preserves a hidden reasoning block for an
                ! assistant continuation after the last user turn.  Earlier
                ! assistant turns contribute visible content, while their
                ! reasoning is intentionally omitted from the next prompt.
                if (i > last_user_index) then
                    call prompt%append('<think>')
                    call prompt%append(char(10))
                    call prompt%append(reasoning)
                    call prompt%append(char(10))
                    call prompt%append('</think>')
                    call prompt%append(char(10))
                    call prompt%append(char(10))
                end if
            end if
            ! Emit the normalized content so legacy inline think markers do not
            ! get fed back into the model as ordinary assistant text.
            call prompt%append_string(content)
            call prompt%append('<|im_end|>')
            call prompt%append(char(10))
        end do
        call prompt%append('<|im_start|>assistant')
        call prompt%append(char(10))
        ! Qwen3's template keeps the thinking block open for generation and
        ! closes it in the prompt when thinking is disabled.  This is the
        ! exact distinction consumed by llama.cpp's Qwen chat template.
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

    subroutine completion_body(model, content, reasoning, tokens, chat, stream, response)
        type(string_t), intent(in) :: model, content, reasoning
        integer, intent(in) :: tokens
        logical, intent(in) :: chat, stream
        type(string_t), intent(out) :: response
        type(string_t) :: model_json, text_json, reasoning_json
        integer :: clock

        call response%clear()
        call system_clock(count=clock)
        model_json = json_escape(model)
        text_json = json_escape(content)
        if (reasoning%length() > 0) reasoning_json = json_escape(reasoning)
        if (chat .and. stream) then
            ! A streaming response is an SSE sequence.  Reasoning and visible
            ! content are separate deltas, matching llama.cpp and clients
            ! that render the hidden channel independently.
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
            call response%append('data: {"id":"fortai-native","object":"chat.completion.chunk","created":')
            call response%append_int(clock)
            call response%append(',"model":')
            call response%append_string(model_json)
            call response%append(',"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"fortai_backend":"fortai"}')
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
            call response%append_string(text_json)
            if (reasoning%length() > 0) then
                call response%append(',"reasoning_content":')
                call response%append_string(reasoning_json)
            end if
            call response%append('},"finish_reason":"stop"}],"usage":{"prompt_tokens":0,"completion_tokens":')
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

    integer(c_int) function fortai_native_http_handle(request, request_length, model, cuda, response, &
            response_capacity, response_length, status, content_type, content_type_capacity) &
            bind(C, name='fortai_native_http_handle')
        character(kind=c_char), intent(in) :: request(*), model(*)
        integer(c_int), value, intent(in) :: request_length, cuda, response_capacity, content_type_capacity
        character(kind=c_char), intent(out) :: response(*), content_type(*)
        integer(c_int), intent(out) :: response_length, status
        type(string_t) :: request_text, model_text, method, path, body, result, prompt, generated
        type(string_t) :: content, reasoning
        type(message_t), allocatable :: messages(:)
        integer :: count, max_tokens, tokens, result_code, required
        integer(int64) :: seed
        real(real32) :: temperature
        logical :: okay, chat, stream, enable_thinking, temperature_valid
        character(len=:), allocatable :: path_value
        character(len=:), allocatable :: reasoning_format

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
        if (method%as_character() /= 'POST' .or. (path_value /= '/v1/chat/completions' .and. &
            path_value /= '/chat/completions' .and. path_value /= '/v1/completions' .and. path_value /= '/completion')) then
            call error_body(404, 'endpoint not available', result); status = 404_c_int
            call copy_result(result, 'application/json', response, response_capacity, response_length, &
                content_type, content_type_capacity, fortai_native_http_handle); return
        end if
        chat = index(path_value, 'chat') > 0
        max_tokens = json_integer(body%as_character(), 'max_tokens', json_integer(body%as_character(), 'max_completion_tokens', 128))
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
        reasoning_format = json_string_value(body%as_character(), 'reasoning_format', 'auto')
        if (chat) then
            if (.not. parse_messages(body%as_character(), messages, count)) then
                call error_body(400, 'messages must contain role/content strings', result); status = 400_c_int
                call copy_result(result, 'application/json', response, response_capacity, response_length, &
                    content_type, content_type_capacity, fortai_native_http_handle); return
            end if
            prompt = format_chat(messages, count, enable_thinking)
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
        if (chat) then
            call split_reasoning(generated%as_character(), enable_thinking, reasoning_format, content, reasoning)
        else
            content = generated
            call reasoning%clear()
        end if
        call completion_body(model_text, content, reasoning, tokens, chat, stream, result)
        status = 200_c_int
        block
            character(len=32) :: mime
            mime = 'application/json'
            if (chat .and. stream) mime = 'text/event-stream'
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
