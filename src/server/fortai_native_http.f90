module fortai_native_http
    !! OpenAI-compatible protocol policy for the native FortAI service.
    !!
    !! The C companion only reads and writes socket bytes.  JSON, chat
    !! templating, response serialization, and the web UI live here so they
    !! share the same growable string_t implementation as the runtime.
    use, intrinsic :: iso_c_binding, only: c_char, c_int, c_null_char
    use, intrinsic :: iso_fortran_env, only: int32, int64, real32, real64
    use fortai_native_service, only: fortai_native_service_complete_text_sampling, &
        fortai_native_service_tokenize, fortai_native_service_detokenize, fortai_native_service_token_piece, &
        fortai_native_service_default_thinking, fortai_native_service_supports_preserve_thinking, &
        fortai_native_service_supports_reasoning_effort, fortai_native_service_mtp_available, &
        fortai_native_service_mtp_active, fortai_native_service_external_draft_active, &
        fortai_native_service_mtp_sidecar_active, fortai_native_service_device_pipeline, &
        fortai_native_service_context_size, fortai_native_service_cache_reuse_supported, &
        fortai_native_service_cache_reuse_active, fortai_native_service_cache_reuse_count, &
        fortai_native_service_last_prompt_ms, fortai_native_service_last_generation_ms, &
        fortai_native_service_last_prompt_tokens, fortai_native_service_last_generation_tokens
    use fortai_whisper_service, only: fortai_whisper_http_handle
    use fortai_string, only: string_t
    implicit none
    private

    integer, parameter :: max_generation = 32768
    integer, parameter :: max_messages = 256
    integer, parameter :: max_token_array = 1000000
    integer(int64), save :: request_count = 0_int64
    integer(int64), save :: generation_count = 0_int64

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
    public :: fortai_native_http_json_integer_checked

    interface
        integer(c_int) function fortai_server_set_environment_http(name, value) &
                bind(C, name='fortai_server_set_environment')
            import c_char, c_int
            character(kind=c_char), intent(in) :: name(*), value(*)
        end function fortai_server_set_environment_http
    end interface

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

    integer function json_top_level_key(text, key)
        character(len=*), intent(in) :: text, key
        type(string_t) :: member, ignored
        integer :: position, object_end, after, colon, value_position, value_after

        json_top_level_key = 0
        position = skip_space(text, 1)
        if (position > len(text)) return
        if (text(position:position) /= '{') return
        object_end = matching_delimiter(text, position, '{', '}')
        if (object_end == 0) return
        position = position + 1
        do while (position < object_end)
            position = skip_space(text, position)
            if (position >= object_end) return
            if (text(position:position) == '}') return
            if (text(position:position) /= '"') return
            if (.not. json_string(text, position, member, after)) return
            colon = skip_space(text, after)
            if (colon >= object_end) return
            if (text(colon:colon) /= ':') return
            value_position = skip_space(text, colon + 1)
            if (value_position >= object_end) return
            if (member%as_character() == key) then
                json_top_level_key = value_position
                return
            end if
            if (.not. json_raw_value(text, value_position, ignored, value_after)) return
            position = skip_space(text, value_after)
            if (position >= object_end) return
            if (text(position:position) == ',') then
                position = position + 1
            else if (text(position:position) == '}') then
                return
            else
                return
            end if
        end do
    end function json_top_level_key

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
        integer :: limit, value, position, object_end, type_value
        logical :: found
        type(string_t) :: parsed, content_type

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

        ! OpenAI content arrays may contain text and image/audio parts.  The
        ! native Qwen text runtime has no vision/audio encoder yet; reject
        ! every non-text part instead of silently dropping it and returning a
        ! misleading text-only answer.  A single object is accepted for
        ! compatibility with clients that send {"type":"text",...}.
        if (text(first:first) == '{') then
            type_value = json_key(text, 'type', first, limit)
            if (type_value > 0) then
                if (.not. json_string(text, type_value, content_type, after)) then
                    json_content = .false.
                    return
                end if
                if (.not. content_type%equals('text')) then
                    json_content = .false.
                    return
                end if
            end if
            value = json_key(text, 'text', first, limit)
            if (value == 0) then
                json_content = .false.
                return
            end if
            if (.not. json_string(text, value, output, after)) then
                json_content = .false.
                return
            end if
            json_content = .true.
            return
        end if

        call output%clear()
        found = .false.
        position = first + 1
        do
            position = skip_space(text, position)
            if (position >= limit) exit
            if (text(position:position) /= '{') then
                json_content = .false.
                return
            end if
            object_end = matching_delimiter(text, position, '{', '}')
            if (object_end == 0 .or. object_end > limit) then
                json_content = .false.
                return
            end if
            type_value = json_key(text, 'type', position, object_end)
            if (type_value > 0) then
                if (.not. json_string(text, type_value, content_type, after)) then
                    json_content = .false.
                    return
                end if
                if (.not. content_type%equals('text')) then
                    json_content = .false.
                    return
                end if
            end if
            value = json_key(text, 'text', position, object_end)
            if (value == 0) then
                json_content = .false.
                return
            end if
            if (.not. json_string(text, value, parsed, after)) then
                json_content = .false.
                return
            end if
            call output%append_string(parsed)
            found = .true.
            position = skip_space(text, object_end + 1)
            if (position >= limit) exit
            if (text(position:position) == ',') then
                position = position + 1
                cycle
            end if
            if (text(position:position) /= ']') then
                json_content = .false.
                return
            end if
            exit
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
        position = json_top_level_key(text, key)
        if (position == 0) return
        if (position > len(text)) return
        if (text(position:position) /= '[') return
        if (.not. json_raw_value(text, position, output, after)) return
        json_array_value = output%length() > 1
    end function json_array_value

    logical function parse_prompt_array(text, array_key, prompts, count, max_count)
        character(len=*), intent(in) :: text, array_key
        type(string_t), allocatable, intent(out) :: prompts(:)
        integer, intent(out) :: count
        integer, intent(in) :: max_count
        integer :: position, limit, after
        type(string_t) :: parsed
        type(string_t), allocatable :: grown(:)

        parse_prompt_array = .false.
        count = 0
        allocate(prompts(0))
        if (max_count <= 0) return
        position = json_key(text, array_key, 1)
        if (position == 0 .or. position > len(text)) return
        if (text(position:position) /= '[') return
        limit = matching_delimiter(text, position, '[', ']')
        if (limit == 0) return
        position = skip_space(text, position + 1)
        do while (position < limit)
            if (text(position:position) /= '"') return
            if (.not. json_string(text, position, parsed, after)) return
            if (parsed%length() == 0 .or. count >= max_count) return
            allocate(grown(count + 1))
            if (count > 0) grown(:count) = prompts(:count)
            grown(count + 1) = parsed
            call move_alloc(grown, prompts)
            count = count + 1
            position = skip_space(text, after)
            if (position > limit) return
            if (text(position:position) == ']') exit
            if (text(position:position) /= ',') return
            position = skip_space(text, position + 1)
        end do
        if (position <= limit) then
            if (text(position:position) == ']') parse_prompt_array = count > 0
        end if
    end function parse_prompt_array

    logical function parse_token_array(text, array_key, ids)
        character(len=*), intent(in) :: text, array_key
        integer(int32), allocatable, intent(out) :: ids(:)
        integer(int32), allocatable :: work(:), grown(:)
        integer :: position, limit, cursor, count, capacity, digit, sign
        integer(int64) :: value, max_token_id
        logical :: saw_digit

        allocate(ids(0))
        parse_token_array = .false.
        position = json_top_level_key(text, array_key)
        if (position == 0 .or. position > len(text)) return
        if (text(position:position) /= '[') return
        limit = matching_delimiter(text, position, '[', ']')
        if (limit == 0) return
        max_token_id = int(huge(0_int32), int64)
        capacity = 64
        allocate(work(capacity))
        count = 0
        cursor = skip_space(text, position + 1)
        if (cursor == limit) then
            deallocate(work)
            parse_token_array = .true.
            return
        end if
        if (cursor > limit) then
            deallocate(work)
            return
        end if
        do
            if (cursor >= limit) then
                deallocate(work)
                return
            end if
            sign = 1
            if (text(cursor:cursor) == '-') then
                sign = -1
                cursor = cursor + 1
            else if (text(cursor:cursor) == '+') then
                deallocate(work)
                return
            end if
            value = 0_int64
            saw_digit = .false.
            do while (cursor < limit)
                if (text(cursor:cursor) < '0' .or. text(cursor:cursor) > '9') exit
                saw_digit = .true.
                digit = iachar(text(cursor:cursor)) - iachar('0')
                if (value > (huge(value) - int(digit, int64)) / 10_int64) then
                    deallocate(work)
                    return
                end if
                value = 10_int64 * value + int(digit, int64)
                cursor = cursor + 1
            end do
            if (.not. saw_digit .or. sign < 0 .or. value > max_token_id) then
                deallocate(work)
                return
            end if
            if (count >= max_token_array) then
                deallocate(work)
                return
            end if
            if (count == capacity) then
                allocate(grown(min(max_token_array, 2 * capacity)))
                grown(:capacity) = work
                call move_alloc(grown, work)
                capacity = size(work)
            end if
            count = count + 1
            work(count) = int(value, int32)
            cursor = skip_space(text, cursor)
            if (cursor == limit) exit
            if (cursor > limit) then
                deallocate(work)
                return
            end if
            if (text(cursor:cursor) /= ',') then
                deallocate(work)
                return
            end if
            cursor = skip_space(text, cursor + 1)
            if (cursor >= limit) then
                deallocate(work)
                return
            end if
        end do
        deallocate(ids)
        allocate(ids(count))
        if (count > 0) ids = work(:count)
        deallocate(work)
        parse_token_array = .true.
    end function parse_token_array

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

    integer function fortai_native_http_json_integer_checked(text, key, fallback, valid, minimum)
        character(len=*), intent(in) :: text, key
        integer, intent(in) :: fallback
        logical, intent(out), optional :: valid
        integer, intent(in), optional :: minimum
        integer :: position, last, value, minimum_value, sign

        fortai_native_http_json_integer_checked = fallback
        if (present(valid)) valid = .true.
        minimum_value = 0
        if (present(minimum)) minimum_value = minimum
        position = json_key(text, key, 1)
        if (position == 0) return
        if (position > len(text)) then
            if (present(valid)) valid = .false.
            return
        end if
        sign = 1
        if (text(position:position) == '-') then
            sign = -1
            position = position + 1
        else if (text(position:position) == '+') then
            position = position + 1
        end if
        last = position
        value = 0
        do while (last <= len(text))
            if (text(last:last) < '0' .or. text(last:last) > '9') exit
            if (value > 100000000) then
                if (present(valid)) valid = .false.
                return
            end if
            value = 10 * value + iachar(text(last:last)) - iachar('0')
            last = last + 1
        end do
        if (last == position) then
            if (present(valid)) valid = .false.
            return
        end if
        if (sign * value < minimum_value) then
            if (present(valid)) valid = .false.
            return
        end if
        last = skip_space(text, last)
        if (last <= len(text)) then
            if (text(last:last) /= ',' .and. text(last:last) /= '}' .and. text(last:last) /= ']') then
                if (present(valid)) valid = .false.
                return
            end if
        end if
        ! Preserve the parsed sign.  In particular, llama.cpp uses
        ! repeat_last_n=-1 to mean "use the complete available history";
        ! returning the unsigned magnitude silently changed that to a one
        ! token window.
        fortai_native_http_json_integer_checked = sign * value
    end function fortai_native_http_json_integer_checked

    integer function json_token_limit(text, responses, valid)
        character(len=*), intent(in) :: text
        logical, intent(in) :: responses
        logical, intent(out) :: valid
        integer :: value

        value = server_integer_default('FORTAI_MAX_TOKENS', -1)
        valid = .true.
        if (responses) then
            if (json_key(text, 'max_output_tokens', 1) > 0) then
                value = fortai_native_http_json_integer_checked(text, 'max_output_tokens', -1, valid)
            else if (json_key(text, 'max_completion_tokens', 1) > 0) then
                value = fortai_native_http_json_integer_checked(text, 'max_completion_tokens', -1, valid)
            else if (json_key(text, 'max_tokens', 1) > 0) then
                value = fortai_native_http_json_integer_checked(text, 'max_tokens', -1, valid)
            end if
        else if (json_key(text, 'max_tokens', 1) > 0) then
            value = fortai_native_http_json_integer_checked(text, 'max_tokens', -1, valid)
        else if (json_key(text, 'max_completion_tokens', 1) > 0) then
            value = fortai_native_http_json_integer_checked(text, 'max_completion_tokens', -1, valid)
        end if
        if (.not. valid) then
            json_token_limit = 0
            return
        end if
        if (value == -1) value = max_generation
        if (value <= 0 .or. value > max_generation) then
            valid = .false.
            json_token_limit = 0
            return
        end if
        json_token_limit = value
    end function json_token_limit

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

    logical function json_boolean_checked(text, key, fallback, valid, found)
        character(len=*), intent(in) :: text, key
        logical, intent(in) :: fallback
        logical, intent(out) :: valid, found
        integer :: position, after

        json_boolean_checked = fallback
        valid = .true.
        found = .false.
        position = json_top_level_key(text, key)
        if (position == 0) return
        found = .true.
        position = skip_space(text, position)
        if (position + 3 <= len(text)) then
            if (text(position:position + 3) == 'true') then
                after = skip_space(text, position + 4)
                if (after > len(text)) then
                    json_boolean_checked = .true.
                    return
                end if
                if (text(after:after) == ',' .or. text(after:after) == '}' .or. &
                    text(after:after) == ']') then
                    json_boolean_checked = .true.
                    return
                end if
            end if
        end if
        if (position + 4 <= len(text)) then
            if (text(position:position + 4) == 'false') then
                after = skip_space(text, position + 5)
                if (after > len(text)) then
                    json_boolean_checked = .false.
                    return
                end if
                if (text(after:after) == ',' .or. text(after:after) == '}' .or. &
                    text(after:after) == ']') then
                    json_boolean_checked = .false.
                    return
                end if
            end if
        end if
        valid = .false.
        json_boolean_checked = fallback
    end function json_boolean_checked

    logical function json_object_boolean_checked(text, object_key, key, fallback, valid, found)
        character(len=*), intent(in) :: text, object_key, key
        logical, intent(in) :: fallback
        logical, intent(out) :: valid, found
        integer :: object_value, object_end

        json_object_boolean_checked = fallback
        valid = .true.
        found = .false.
        object_value = json_top_level_key(text, object_key)
        if (object_value == 0) return
        if (object_value > len(text)) then
            valid = .false.
            return
        end if
        if (text(object_value:object_value) /= '{') then
            valid = .false.
            return
        end if
        object_end = matching_delimiter(text, object_value, '{', '}')
        if (object_end == 0) then
            valid = .false.
            return
        end if
        json_object_boolean_checked = json_boolean_checked(text(object_value:object_end), key, fallback, valid, found)
    end function json_object_boolean_checked

    logical function json_object_boolean(text, object_key, key, fallback)
        character(len=*), intent(in) :: text, object_key, key
        logical, intent(in) :: fallback
        integer :: object_value, object_end

        json_object_boolean = fallback
        object_value = json_top_level_key(text, object_key)
        if (object_value == 0 .or. object_value > len(text)) return
        if (text(object_value:object_value) /= '{') return
        object_end = matching_delimiter(text, object_value, '{', '}')
        if (object_end == 0) return
        json_object_boolean = json_boolean(text(object_value:object_end), key, fallback)
    end function json_object_boolean

    function normalize_reasoning_effort(value, valid) result(normalized)
        character(len=*), intent(in) :: value
        logical, intent(out) :: valid
        character(len=:), allocatable :: normalized

        valid = .true.
        select case (trim(value))
        case ('')
            normalized = ''
        case ('default')
            normalized = ''
        case ('none', 'off')
            normalized = 'off'
        case ('minimal', 'low')
            normalized = 'low'
        case ('medium')
            normalized = 'medium'
        case ('high', 'xhigh', 'max')
            normalized = 'xhigh'
        case default
            normalized = ''
            valid = .false.
        end select
    end function normalize_reasoning_effort

    function json_string_checked(text, key, fallback, valid, found) result(value)
        character(len=*), intent(in) :: text, key, fallback
        logical, intent(out) :: valid, found
        character(len=:), allocatable :: value
        type(string_t) :: parsed
        integer :: position, after

        call parsed%set(fallback)
        valid = .true.
        found = .false.
        position = json_top_level_key(text, key)
        if (position == 0) then
            value = parsed%as_character()
            return
        end if
        found = .true.
        if (.not. json_string(text, position, parsed, after)) then
            valid = .false.
            call parsed%set(fallback)
        end if
        value = parsed%as_character()
    end function json_string_checked

    function json_object_string_checked(text, object_key, key, fallback, valid, found) result(value)
        character(len=*), intent(in) :: text, object_key, key, fallback
        logical, intent(out) :: valid, found
        character(len=:), allocatable :: value
        type(string_t) :: parsed
        integer :: object_value, object_end, position, after

        call parsed%set(fallback)
        valid = .true.
        found = .false.
        object_value = json_top_level_key(text, object_key)
        if (object_value == 0) then
            value = parsed%as_character()
            return
        end if
        if (object_value > len(text)) then
            valid = .false.
            value = parsed%as_character()
            return
        end if
        if (text(object_value:object_value) /= '{') then
            valid = .false.
            value = parsed%as_character()
            return
        end if
        object_end = matching_delimiter(text, object_value, '{', '}')
        if (object_end == 0) then
            valid = .false.
            value = parsed%as_character()
            return
        end if
        position = json_top_level_key(text(object_value:object_end), key)
        if (position > 0) then
            found = .true.
            if (.not. json_string(text(object_value:object_end), position, parsed, after)) then
                valid = .false.
                call parsed%set(fallback)
            end if
        end if
        value = parsed%as_character()
    end function json_object_string_checked

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

    function format_chat(messages, count, enable_thinking, tools_json, reasoning_instruction, &
            preserve_thinking, add_generation_prompt) result(prompt)
        type(message_t), intent(in) :: messages(:)
        integer, intent(in) :: count
        logical, intent(in) :: enable_thinking
        type(string_t), intent(in) :: tools_json
        type(string_t), intent(in) :: reasoning_instruction
        logical, intent(in) :: preserve_thinking
        logical, intent(in), optional :: add_generation_prompt
        type(string_t) :: prompt
        integer :: i, start_index, last_user_index, leading_system_count
        character(len=:), allocatable :: role, reasoning, raw_content
        type(string_t) :: content, parsed_reasoning, merged_system
        logical :: tool_group_open, has_real_reasoning, append_generation_prompt

        call prompt%clear()
        append_generation_prompt = .true.
        if (present(add_generation_prompt)) append_generation_prompt = add_generation_prompt
        call merged_system%clear()
        leading_system_count = 0
        do while (leading_system_count < count)
            if (messages(leading_system_count + 1)%role%as_character() /= 'system') exit
            leading_system_count = leading_system_count + 1
            raw_content = messages(leading_system_count)%content%as_character()
            if (has_nonblank_text(raw_content)) then
                if (merged_system%length() > 0) call merged_system%append(char(10))
                call append_trimmed_text(merged_system, raw_content)
            end if
        end do
        last_user_index = 0
        do i = 1, count
            if (messages(i)%role%as_character() == 'user') then
                raw_content = messages(i)%content%as_character()
                if (.not. is_tool_response(raw_content)) last_user_index = i
            end if
        end do

        start_index = 1
        if (has_tool_entries(tools_json)) then
            call append_tool_system(prompt, tools_json, merged_system, merged_system%length() > 0, reasoning_instruction)
            start_index = leading_system_count + 1
        else if (merged_system%length() > 0 .or. reasoning_instruction%length() > 0) then
            call prompt%append('<|im_start|>system')
            call prompt%append(char(10))
            if (reasoning_instruction%length() > 0) then
                call append_trimmed_text(prompt, reasoning_instruction%as_character())
                if (merged_system%length() > 0) then
                    call prompt%append(char(10))
                    call prompt%append(char(10))
                end if
            end if
            if (merged_system%length() > 0) call prompt%append_string(merged_system)
            call prompt%append('<|im_end|>')
            call prompt%append(char(10))
            start_index = leading_system_count + 1
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
                call append_trimmed_text(prompt, messages(i)%content%as_character())
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
                raw_content = content%as_character()
                call content%clear()
                call append_trimmed_text(content, raw_content)
                has_real_reasoning = has_nonblank_text(reasoning)
                ! Qwen3.5 keeps reasoning only for assistant turns after the
                ! latest real user query (including multi-step tool turns).
                if (preserve_thinking .or. i > last_user_index) then
                    ! Qwen3.8 preserves the trace in every historical turn;
                    ! older Qwen templates only need a block for a real trace
                    ! after the latest user query.  Avoid injecting empty
                    ! historical <think> blocks into those older prompts.
                    if (preserve_thinking .or. has_real_reasoning) then
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
                else
                    call prompt%append_string(content)
                end if
                if (messages(i)%tool_calls%length() > 0) then
                    call append_assistant_tool_calls(prompt, messages(i)%tool_calls, content%length() > 0)
                end if
            else
                call append_trimmed_text(prompt, content%as_character())
            end if
            call prompt%append('<|im_end|>')
            call prompt%append(char(10))
        end do
        if (tool_group_open) then
            call prompt%append('<|im_end|>')
            call prompt%append(char(10))
        end if
        if (append_generation_prompt) then
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

    logical function has_nonblank_text(text)
        character(len=*), intent(in) :: text
        integer :: first, last

        call trim_bounds(text, first, last)
        has_nonblank_text = first <= last
    end function has_nonblank_text

    subroutine append_tool_system(prompt, tools_json, system_content, has_system_content, reasoning_instruction)
        type(string_t), intent(inout) :: prompt
        type(string_t), intent(in) :: tools_json, system_content
        logical, intent(in) :: has_system_content
        type(string_t), intent(in), optional :: reasoning_instruction

        call prompt%append('<|im_start|>system')
        call prompt%append(char(10))
        if (present(reasoning_instruction)) then
            if (reasoning_instruction%length() > 0) then
                call append_trimmed_text(prompt, reasoning_instruction%as_character())
                call prompt%append(char(10))
                call prompt%append(char(10))
            end if
        end if
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

    integer function utf8_piece_width(text, position)
        character(len=*), intent(in) :: text
        integer, intent(in) :: position
        integer :: first, second, third, fourth, codepoint

        utf8_piece_width = 0
        if (position < 1 .or. position > len(text)) return
        first = iachar(text(position:position))
        if (first < 128) then
            utf8_piece_width = 1
            return
        end if
        if (first >= int(z'c2') .and. first <= int(z'df')) then
            if (position + 1 > len(text)) return
            second = iachar(text(position + 1:position + 1))
            if (second >= int(z'80') .and. second <= int(z'bf')) utf8_piece_width = 2
            return
        end if
        if (first >= int(z'e0') .and. first <= int(z'ef')) then
            if (position + 2 > len(text)) return
            second = iachar(text(position + 1:position + 1))
            third = iachar(text(position + 2:position + 2))
            if (second < int(z'80') .or. second > int(z'bf') .or. third < int(z'80') .or. third > int(z'bf')) return
            codepoint = iand(first, 15) * 4096 + iand(second, 63) * 64 + iand(third, 63)
            if (codepoint < int(z'800') .or. (codepoint >= int(z'd800') .and. codepoint <= int(z'dfff'))) return
            utf8_piece_width = 3
            return
        end if
        if (first >= int(z'f0') .and. first <= int(z'f4')) then
            if (position + 3 > len(text)) return
            second = iachar(text(position + 1:position + 1))
            third = iachar(text(position + 2:position + 2))
            fourth = iachar(text(position + 3:position + 3))
            if (second < int(z'80') .or. second > int(z'bf') .or. third < int(z'80') .or. third > int(z'bf') .or. &
                fourth < int(z'80') .or. fourth > int(z'bf')) return
            codepoint = iand(first, 7) * 262144 + iand(second, 63) * 4096 + iand(third, 63) * 64 + iand(fourth, 63)
            if (codepoint < int(z'10000') .or. codepoint > int(z'10ffff')) return
            utf8_piece_width = 4
        end if
    end function utf8_piece_width

    logical function valid_utf8_piece(text)
        character(len=*), intent(in) :: text
        integer :: position, width

        valid_utf8_piece = .true.
        position = 1
        do while (position <= len(text))
            width = utf8_piece_width(text, position)
            if (width == 0) then
                valid_utf8_piece = .false.
                return
            end if
            position = position + width
        end do
    end function valid_utf8_piece

    subroutine append_token_piece_json(output, piece)
        type(string_t), intent(inout) :: output
        character(len=*), intent(in) :: piece
        type(string_t) :: piece_string, escaped
        integer :: i, byte_value

        call piece_string%set(piece)
        if (valid_utf8_piece(piece)) then
            escaped = json_escape(piece_string)
            call output%append_string(escaped)
            return
        end if
        call output%append('[')
        do i = 1, len(piece)
            if (i > 1) call output%append(',')
            byte_value = iand(iachar(piece(i:i)), int(z'ff'))
            call output%append_int(byte_value)
        end do
        call output%append(']')
    end subroutine append_token_piece_json

    function tokenization_response(ids, with_pieces) result(output)
        integer(int32), intent(in) :: ids(:)
        logical, intent(in) :: with_pieces
        type(string_t) :: output
        character(len=:), allocatable :: piece
        logical :: piece_ok
        integer :: i

        call output%set('{"tokens":[')
        do i = 1, size(ids)
            if (i > 1) call output%append(',')
            if (.not. with_pieces) then
                call output%append_int(int(ids(i)))
            else
                call output%append('{"id":')
                call output%append_int(int(ids(i)))
                call output%append(',"piece":')
                piece_ok = fortai_native_service_token_piece(ids(i), piece)
                if (piece_ok) then
                    call append_token_piece_json(output, piece)
                else
                    call output%append('""')
                end if
                call output%append('}')
            end if
        end do
        call output%append(']}')
        call output%append(char(10))
    end function tokenization_response

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
        associate (ignored_status => status)
        end associate
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
        type(string_t) :: api_prefix_json, api_prefix_text
        character(len=:), allocatable :: api_prefix
        api_prefix = server_api_prefix()
        call api_prefix_text%set(api_prefix)
        api_prefix_json = json_escape(api_prefix_text)
        call page%append('<!doctype html><html lang="en"><head><meta charset="utf-8">')
        call page%append('<meta name="viewport" content="width=device-width,initial-scale=1">')
        call page%append('<title>FortAI</title><style>')
        call page%append(':root{color-scheme:dark;font:16px system-ui,sans-serif}body{margin:0;background:#10131a;color:#edf2f7}')
        call page%append('main{max-width:960px;margin:auto;padding:28px}.brand{display:flex;align-items:center;gap:12px}')
        call page%append('.dot{width:13px;height:13px;border-radius:50%;background:#53e6a7;box-shadow:0 0 18px #53e6a7}')
        call page%append('h1{font-size:1.5rem;margin:0}.sub{color:#9ba7b5;margin:4px 0 24px}')
        call page%append('.chat{min-height:55vh;border:1px solid #2d3748')
        call page%append(';border-radius:14px;padding:18px;background:#151a23;overflow:auto}')
        call page%append('.msg{white-space:pre-wrap;line-height:1.5;margin:12px 0;padding:12px 15px;border-radius:10px;')
        call page%append('max-width:85%}')
        call page%append('.user{margin-left:auto;background:#1d4d43}.assistant{background:#222b39}.role{font-size:.75rem;')
        call page%append('color:#9ba7b5;text-transform:uppercase;letter-spacing:.08em;margin-bottom:5px}')
        call page%append('.thought{margin-bottom:10px;color:#aeb8c5;font-size:.9rem}.thought summary{cursor:pointer;')
        call page%append('color:#73d6b1;margin-bottom:6px}')
        call page%append('form{display:flex;gap:10px;margin-top:14px}textarea{flex:1;resize:vertical;min-height:58px;')
        call page%append('max-height:240px;border:1px solid #39475a;border-radius:10px;padding:12px;background:#0e1117;')
        call page%append('color:inherit;font:inherit}')
        call page%append('button{border:0;border-radius:10px;padding:0 22px;background:#53e6a7;color:#0b2118;')
        call page%append('font-weight:700;cursor:pointer}.status{color:#9ba7b5;font-size:.85rem;margin-top:9px}</style>')
        call page%append('</head><body><main>')
        call page%append('<div class=brand><span class=dot></span><div><h1>FortAI</h1>')
        call page%append('<div class=sub>Native Qwen3.5 CUDA inference</div></div></div>')
        call page%append('<section id=chat class="chat"><div class="msg assistant"><div class=role>FortAI</div>Ready.</div>')
        call page%append('</section>')
        call page%append('<form id=form><textarea id=input placeholder="Message FortAI…"></textarea>')
        call page%append('<button id=send>Send</button>')
        call page%append('</form>')
        call page%append('<div id=status class=status>Connected to the FortAI-native server.</div></main><script>')
        call page%append('const apiPrefix=')
        call page%append_string(api_prefix_json)
        call page%append(',chat=document.querySelector("#chat"),input=document.querySelector("#input"),')
        call page%append('form=document.querySelector("#form"),')
        call page%append('send=document.querySelector("#send"),status=document.querySelector("#status"),messages=[];')
        call page%append('function add(role,text){const el=document.createElement("div");el.className="msg "+role;')
        call page%append('const r=document.createElement("div");r.className="role";r.textContent=role==="user"?"You":"FortAI";')
        call page%append('const t=document.createElement("div");t.textContent=text;')
        call page%append('el.append(r,t);chat.append(el);chat.scrollTop=chat.scrollHeight;return t}')
        call page%append('form.addEventListener("submit",async e=>{e.preventDefault();const text=input.value.trim();')
        call page%append('if(!text)return;input.value="";add("user",text);messages.push({role:"user",content:text});')
        call page%append('send.disabled=true;status.textContent="Generating…";const target=add("assistant","");')
        call page%append('try{const r=await fetch(apiPrefix+"/v1/chat/completions",{method:"POST",')
        call page%append('headers:{"content-type":"application/json"},body:JSON.stringify({model:"qwen",messages,')
        call page%append('max_tokens:512,stream:false,temperature:0})});const d=await r.json();')
        call page%append('if(!r.ok)throw Error(d.error?.message||"request failed");const m=d.choices?.[0]?.message||{};')
        call page%append('target.textContent=m.content||"";')
        call page%append('if(m.reasoning_content){const details=document.createElement("details"),')
        call page%append('summary=document.createElement("summary"),thought=document.createElement("div");')
        call page%append('details.className="thought";')
        call page%append('summary.textContent="Thinking";thought.textContent=m.reasoning_content;details.append(summary,thought);')
        call page%append('target.parentElement.insertBefore(details,target)}messages.push({role:"assistant",content:m.content||"",')
        call page%append('reasoning_content:m.reasoning_content||""});status.textContent="FortAI native CUDA"}')
        call page%append('catch(err){target.textContent="Error: "+err.message;status.textContent="Request failed"}')
        call page%append('finally{send.disabled=false;input.focus()}})</script></body></html>')
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

    subroutine append_json_real(output, value)
        type(string_t), intent(inout) :: output
        real(real64), intent(in) :: value
        character(len=64) :: number

        write(number, '(f0.3)') value
        call output%append(trim(adjustl(number)))
    end subroutine append_json_real

    subroutine append_timings(output, prompt_tokens, generation_tokens)
        type(string_t), intent(inout) :: output
        integer, intent(in) :: prompt_tokens, generation_tokens
        real(real64) :: prompt_ms, generation_ms, prompt_rate, generation_rate
        integer(int64) :: measured_prompt_tokens, measured_generation_tokens

        prompt_ms = max(0.0_real64, fortai_native_service_last_prompt_ms())
        generation_ms = max(0.0_real64, fortai_native_service_last_generation_ms())
        measured_prompt_tokens = max(0_int64, fortai_native_service_last_prompt_tokens())
        measured_generation_tokens = max(0_int64, fortai_native_service_last_generation_tokens())
        if (measured_prompt_tokens == 0_int64 .and. prompt_tokens > 0 .and. prompt_ms > 0.0_real64) then
            measured_prompt_tokens = int(prompt_tokens, int64)
        end if
        if (measured_generation_tokens == 0_int64 .and. generation_tokens > 0 .and. generation_ms > 0.0_real64) then
            measured_generation_tokens = int(generation_tokens, int64)
        end if
        prompt_rate = 0.0_real64
        generation_rate = 0.0_real64
        if (prompt_ms > 0.0_real64) prompt_rate = 1000.0_real64 * real(measured_prompt_tokens, real64) / prompt_ms
        if (generation_ms > 0.0_real64) generation_rate = &
            1000.0_real64 * real(measured_generation_tokens, real64) / generation_ms
        call output%append('"timings":{"prompt_n":')
        call output%append_int(int(measured_prompt_tokens))
        call output%append(',"prompt_ms":')
        call append_json_real(output, prompt_ms)
        call output%append(',"prompt_per_second":')
        call append_json_real(output, prompt_rate)
        call output%append(',"predicted_n":')
        call output%append_int(int(measured_generation_tokens))
        call output%append(',"predicted_ms":')
        call append_json_real(output, generation_ms)
        call output%append(',"predicted_per_second":')
        call append_json_real(output, generation_rate)
        call output%append('}')
    end subroutine append_timings

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

    subroutine completion_body(model, content, reasoning, prompt_tokens, tokens, chat, stream, response, &
            tool_calls, tool_count)
        type(string_t), intent(in) :: model, content, reasoning
        type(tool_call_t), intent(in) :: tool_calls(:)
        integer, intent(in) :: prompt_tokens, tokens, tool_count
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
            call response%append('}],')
            call append_timings(response, prompt_tokens, tokens)
            call response%append(',"fortai_backend":"fortai"}')
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
            call response%append('}],"usage":{"prompt_tokens":')
            call response%append_int(prompt_tokens)
            call response%append(',"completion_tokens":')
            call response%append_int(tokens); call response%append(',"total_tokens":')
            call response%append_int(prompt_tokens + tokens)
            call response%append('},')
            call append_timings(response, prompt_tokens, tokens)
            call response%append(',"fortai_backend":"fortai"}'); call response%append(char(10))
        else
            call response%append(',"choices":[{"index":0,"text":')
            call response%append_string(text_json)
            call response%append(',"finish_reason":"stop"}],"usage":{"prompt_tokens":')
            call response%append_int(prompt_tokens)
            call response%append(',"completion_tokens":')
            call response%append_int(tokens); call response%append(',"total_tokens":')
            call response%append_int(prompt_tokens + tokens)
            call response%append('},')
            call append_timings(response, prompt_tokens, tokens)
            call response%append(',"fortai_backend":"fortai"}'); call response%append(char(10))
        end if
    end subroutine completion_body

    subroutine completion_batch_body(model, contents, prompt_tokens, tokens, response)
        type(string_t), intent(in) :: model, contents(:)
        integer, intent(in) :: prompt_tokens(:), tokens(:)
        type(string_t), intent(out) :: response
        integer :: clock, i, prompt_total, token_total

        call response%clear()
        clock = unix_timestamp()
        prompt_total = 0
        token_total = 0
        call response%append('{"id":"fortai-native","object":"text_completion","created":')
        call response%append_int(clock)
        call response%append(',"model":')
        call response%append_string(json_escape(model))
        call response%append(',"choices":[')
        do i = 1, size(contents)
            if (i > 1) call response%append(',')
            call response%append('{"index":')
            call response%append_int(i - 1)
            call response%append(',"text":')
            call response%append_string(json_escape(contents(i)))
            call response%append(',"finish_reason":"stop"}')
            prompt_total = prompt_total + prompt_tokens(i)
            token_total = token_total + tokens(i)
        end do
        call response%append('],"usage":{"prompt_tokens":')
        call response%append_int(prompt_total)
        call response%append(',"completion_tokens":')
        call response%append_int(token_total)
        call response%append(',"total_tokens":')
        call response%append_int(prompt_total + token_total)
        call response%append('},"fortai_backend":"fortai"}')
        call response%append(char(10))
    end subroutine completion_batch_body

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

    function response_json(model, content, reasoning, prompt_tokens, tokens, tool_calls, tool_count) result(response)
        type(string_t), intent(in) :: model, content, reasoning
        type(tool_call_t), intent(in) :: tool_calls(:)
        integer, intent(in) :: prompt_tokens, tokens, tool_count
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
        call response%append('],"input":[],"instructions":null,"usage":{"input_tokens":')
        call response%append_int(prompt_tokens)
        call response%append(',"output_tokens":')
        call response%append_int(tokens)
        call response%append(',"total_tokens":')
        call response%append_int(prompt_tokens + tokens)
        call response%append('},"store":false,')
        call append_timings(response, prompt_tokens, tokens)
        call response%append(',"fortai_backend":"fortai"}')
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

    function response_stream_json(model, content, reasoning, prompt_tokens, tokens, tool_calls, tool_count) result(response)
        type(string_t), intent(in) :: model, content, reasoning
        type(tool_call_t), intent(in) :: tool_calls(:)
        integer, intent(in) :: prompt_tokens, tokens, tool_count
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

        final_response = response_json(model, content, reasoning, prompt_tokens, tokens, tool_calls, tool_count)
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
        type(string_t) :: request_text, model_text, model_path_text, method, path, body, result, prompt, generated
        type(string_t) :: content, reasoning, tools_json, clean_content, reasoning_instruction
        type(string_t), allocatable :: prompt_batch(:), batch_generated(:)
        type(message_t), allocatable :: messages(:)
        type(tool_call_t), allocatable :: tool_calls(:)
        integer(int32), allocatable :: token_ids(:)
        integer, allocatable :: batch_prompt_tokens(:), batch_tokens(:)
        integer :: count, max_tokens, prompt_tokens, tokens, result_code, required, tool_count
        integer :: prompt_batch_count, batch_limit, batch_index
        integer :: top_k, repeat_last_n
        integer(int64) :: seed
        real(real32) :: temperature, top_p, min_p, repeat_penalty, presence_penalty, frequency_penalty
        logical :: okay, chat, responses, stream, enable_thinking, preserve_thinking
        logical :: reasoning_effort_valid, reasoning_effort_type_valid, reasoning_effort_found
        logical :: nested_reasoning_effort_valid, nested_reasoning_effort_found
        logical :: response_reasoning_effort_valid, response_reasoning_effort_found
        logical :: supports_reasoning_effort, supports_preserve_thinking
        logical :: thinking_explicit, enable_thinking_valid, nested_thinking_valid, nested_thinking_found
        logical :: preserve_thinking_valid, preserve_thinking_found
        logical :: reasoning_format_valid, reasoning_format_found
        logical :: temperature_valid, max_tokens_valid
        logical :: top_p_valid, min_p_valid, repeat_penalty_valid, presence_penalty_valid
        logical :: frequency_penalty_valid, top_k_valid, repeat_last_n_valid, sampling_valid
        logical :: prompt_array, whisper_model
        character(len=:), allocatable :: path_value, model_path
        character(len=:), allocatable :: reasoning_format, reasoning_effort
        integer :: reasoning_budget

        response_length = 0_c_int
        status = 500_c_int
        prompt_batch_count = 0
        prompt_array = .false.
        request_count = request_count + 1_int64
        call request_text%from_c(request, int(request_length))
        call model_path_text%from_c(model)
        model_path = model_path_text%as_character()
        whisper_model = is_whisper_model_path(model_path)
        if (whisper_model) then
            fortai_native_http_handle = fortai_whisper_http_handle(request, request_length, model, cuda, response, &
                response_capacity, response_length, status, content_type, content_type_capacity)
            return
        end if
        model_text = model_path_text
        call server_model_id(model_text)
        call parse_http_request(request_text, method, path, body, okay)
        if (.not. okay) then
            call error_body(400, 'malformed HTTP request', result); status = 400_c_int
            call copy_result(result, 'application/json', response, response_capacity, response_length, &
                content_type, content_type_capacity, fortai_native_http_handle); return
        end if
        path_value = path%as_character()
        call apply_api_prefix(path_value)
        if (path_value == '/metrics' .and. method%as_character() /= 'POST') then
            call metrics_body(result, cuda); status = 200_c_int
            call copy_result(result, 'text/plain; version=0.0.4', response, response_capacity, response_length, &
                content_type, content_type_capacity, fortai_native_http_handle); return
        end if
        if (method%as_character() == 'GET' .and. len_trim(server_text_default('FORTAI_STATIC_PATH', '')) > 0) then
            block
                character(len=64) :: static_mime
                logical :: static_found
                call server_static_file(path_value, result, static_mime, static_found)
                if (static_found) then
                    status = 200_c_int
                    call copy_result(result, static_mime(:len_trim(static_mime)), response, response_capacity, response_length, &
                        content_type, content_type_capacity, fortai_native_http_handle)
                    return
                end if
            end block
        end if
        if (path_value == '/' .or. path_value == '/ui' .or. path_value == '/index.html') then
            if (.not. server_web_ui_enabled()) then
                call error_body(404, 'web UI is disabled', result); status = 404_c_int
                call copy_result(result, 'application/json', response, response_capacity, response_length, &
                    content_type, content_type_capacity, fortai_native_http_handle); return
            end if
            result = web_ui(); status = 200_c_int
            call copy_result(result, 'text/html; charset=utf-8', response, response_capacity, response_length, &
                content_type, content_type_capacity, fortai_native_http_handle); return
        end if
        if (path_value == '/health' .or. path_value == '/v1/health') then
            call result%append('{"status":"ok","backend":"fortai","engine":"fortai-native-qwen35","cuda":')
            call result%append_logical(cuda /= 0_c_int)
            call result%append(',"device_pipeline":')
            call result%append_logical(fortai_native_service_device_pipeline())
            call result%append(',"mtp_available":')
            call result%append_logical(fortai_native_service_mtp_available())
            call result%append(',"mtp_active":')
            call result%append_logical(fortai_native_service_mtp_active())
            call result%append(',"external_draft_active":')
            call result%append_logical(fortai_native_service_external_draft_active())
            call result%append(',"mtp_sidecar_active":')
            call result%append_logical(fortai_native_service_mtp_sidecar_active())
            call result%append(',"service":"fortai-server","settings":')
            call server_settings_json(result)
            call result%append('}'); call result%append(char(10)); status = 200_c_int
            call copy_result(result, 'application/json', response, response_capacity, response_length, &
                content_type, content_type_capacity, fortai_native_http_handle); return
        end if
        if (path_value == '/props' .and. method%as_character() == 'GET') then
            call server_props_json(model_text, model_path_text, result)
            call result%append(char(10)); status = 200_c_int
            call copy_result(result, 'application/json', response, response_capacity, response_length, &
                content_type, content_type_capacity, fortai_native_http_handle); return
        end if
        if (path_value == '/props' .and. method%as_character() == 'POST') then
            if (.not. server_boolean_default('FORTAI_ENDPOINT_PROPS', .false.)) then
                call error_body(404, 'props endpoint disabled', result); status = 404_c_int
            else if (.not. server_props_update(body%as_character(), result)) then
                status = 400_c_int
            else
                call server_props_json(model_text, model_path_text, result); status = 200_c_int
            end if
            call result%append(char(10))
            call copy_result(result, 'application/json', response, response_capacity, response_length, &
                content_type, content_type_capacity, fortai_native_http_handle); return
        end if
        if (path_value == '/slots' .and. method%as_character() == 'GET') then
            if (.not. server_boolean_default('FORTAI_ENDPOINT_SLOTS', .true.)) then
                call error_body(404, 'slots endpoint disabled', result); status = 404_c_int
            else
                call server_slots_json(result); status = 200_c_int
            end if
            call result%append(char(10))
            call copy_result(result, 'application/json', response, response_capacity, response_length, &
                content_type, content_type_capacity, fortai_native_http_handle); return
        end if
        if (path_value == '/tokenize' .and. method%as_character() == 'POST') then
            block
                character(len=:), allocatable :: body_text, content_text
                type(string_t) :: parsed_content
                integer :: content_value, after
                logical :: content_valid, add_special, parse_special, with_pieces
                logical :: option_valid, option_found, token_ok

                body_text = body%as_character()
                content_value = json_top_level_key(body_text, 'content')
                content_valid = .false.
                if (content_value > 0 .and. content_value <= len(body_text)) then
                    if (body_text(content_value:content_value) == '"') then
                        content_valid = json_string(body_text, content_value, parsed_content, after)
                        if (content_valid) content_text = parsed_content%as_character()
                    end if
                end if
                if (.not. content_valid) then
                    call error_body(400, 'content must be a string', result)
                    status = 400_c_int
                    call copy_result(result, 'application/json', response, response_capacity, response_length, &
                        content_type, content_type_capacity, fortai_native_http_handle)
                    return
                end if
                add_special = json_boolean_checked(body_text, 'add_special', .false., option_valid, option_found)
                if (.not. option_valid) then
                    call error_body(400, 'add_special must be a boolean', result)
                    status = 400_c_int
                    call copy_result(result, 'application/json', response, response_capacity, response_length, &
                        content_type, content_type_capacity, fortai_native_http_handle)
                    return
                end if
                parse_special = json_boolean_checked(body_text, 'parse_special', .true., option_valid, option_found)
                if (.not. option_valid) then
                    call error_body(400, 'parse_special must be a boolean', result)
                    status = 400_c_int
                    call copy_result(result, 'application/json', response, response_capacity, response_length, &
                        content_type, content_type_capacity, fortai_native_http_handle)
                    return
                end if
                with_pieces = json_boolean_checked(body_text, 'with_pieces', .false., option_valid, option_found)
                if (.not. option_valid) then
                    call error_body(400, 'with_pieces must be a boolean', result)
                    status = 400_c_int
                    call copy_result(result, 'application/json', response, response_capacity, response_length, &
                        content_type, content_type_capacity, fortai_native_http_handle)
                    return
                end if
                token_ok = fortai_native_service_tokenize(content_text, add_special, parse_special, token_ids)
                if (.not. token_ok) then
                    call error_body(500, 'FortAI tokenization failed', result)
                    status = 500_c_int
                else
                    result = tokenization_response(token_ids, with_pieces)
                    status = 200_c_int
                end if
            end block
            call copy_result(result, 'application/json', response, response_capacity, response_length, &
                content_type, content_type_capacity, fortai_native_http_handle)
            return
        end if
        if (path_value == '/detokenize' .and. method%as_character() == 'POST') then
            block
                character(len=:), allocatable :: decoded_text
                type(string_t) :: decoded_string, escaped
                logical :: token_ok

                if (.not. parse_token_array(body%as_character(), 'tokens', token_ids)) then
                    call error_body(400, 'tokens must be an array of non-negative token IDs', result)
                    status = 400_c_int
                else
                    token_ok = fortai_native_service_detokenize(token_ids, decoded_text)
                    if (.not. token_ok) then
                        call error_body(400, 'tokens contains an invalid token ID', result)
                        status = 400_c_int
                    else
                        call decoded_string%set(decoded_text)
                        escaped = json_escape(decoded_string)
                        call result%set('{"content":')
                        call result%append_string(escaped)
                        call result%append('}')
                        call result%append(char(10))
                        status = 200_c_int
                    end if
                end if
            end block
            call copy_result(result, 'application/json', response, response_capacity, response_length, &
                content_type, content_type_capacity, fortai_native_http_handle)
            return
        end if
        if (path_value == '/apply-template' .and. method%as_character() == 'POST') then
            block
                character(len=:), allocatable :: body_text
                logical :: option_valid, option_found, nested_valid, nested_found
                logical :: add_generation_prompt, enable_thinking, preserve_template_thinking

                body_text = body%as_character()
                add_generation_prompt = json_boolean_checked(body_text, 'add_generation_prompt', .true., &
                    option_valid, option_found)
                if (.not. option_valid) then
                    call error_body(400, 'add_generation_prompt must be a boolean', result)
                    status = 400_c_int
                    call copy_result(result, 'application/json', response, response_capacity, response_length, &
                        content_type, content_type_capacity, fortai_native_http_handle)
                    return
                end if
                enable_thinking = json_boolean_checked(body_text, 'enable_thinking', &
                    fortai_native_service_default_thinking(), option_valid, option_found)
                if (.not. option_valid) then
                    call error_body(400, 'enable_thinking must be a boolean', result)
                    status = 400_c_int
                    call copy_result(result, 'application/json', response, response_capacity, response_length, &
                        content_type, content_type_capacity, fortai_native_http_handle)
                    return
                end if
                enable_thinking = json_object_boolean_checked(body_text, 'chat_template_kwargs', 'enable_thinking', &
                    enable_thinking, nested_valid, nested_found)
                if (.not. nested_valid) then
                    call error_body(400, 'chat_template_kwargs.enable_thinking must be a boolean', result)
                    status = 400_c_int
                    call copy_result(result, 'application/json', response, response_capacity, response_length, &
                        content_type, content_type_capacity, fortai_native_http_handle)
                    return
                end if
                preserve_template_thinking = server_boolean_default('FORTAI_REASONING_PRESERVE', &
                    fortai_native_service_supports_preserve_thinking())
                preserve_template_thinking = json_boolean_checked(body_text, 'preserve_thinking', &
                    preserve_template_thinking, option_valid, option_found)
                if (.not. option_valid) then
                    call error_body(400, 'preserve_thinking must be a boolean', result)
                    status = 400_c_int
                    call copy_result(result, 'application/json', response, response_capacity, response_length, &
                        content_type, content_type_capacity, fortai_native_http_handle)
                    return
                end if
                preserve_template_thinking = json_object_boolean_checked(body_text, 'chat_template_kwargs', &
                    'preserve_thinking', preserve_template_thinking, nested_valid, nested_found)
                if (.not. nested_valid) then
                    call error_body(400, 'chat_template_kwargs.preserve_thinking must be a boolean', result)
                    status = 400_c_int
                    call copy_result(result, 'application/json', response, response_capacity, response_length, &
                        content_type, content_type_capacity, fortai_native_http_handle)
                    return
                end if
                call tools_json%clear()
                if (.not. json_array_value(body_text, 'tools', tools_json)) call tools_json%clear()
                call reasoning_instruction%clear()
                if (.not. parse_messages(body_text, messages, count)) then
                    call error_body(400, 'messages must contain text-only role/content strings; native multimodal ' // &
                        'image/audio input is unsupported', result)
                    status = 400_c_int
                else
                    prompt = format_chat(messages, count, enable_thinking, tools_json, reasoning_instruction, &
                        preserve_template_thinking, add_generation_prompt)
                    call result%set('{"prompt":')
                    call result%append_string(json_escape(prompt))
                    call result%append('}')
                    call result%append(char(10))
                    status = 200_c_int
                end if
            end block
            call copy_result(result, 'application/json', response, response_capacity, response_length, &
                content_type, content_type_capacity, fortai_native_http_handle)
            return
        end if
        if ((path_value == '/v1/models' .or. index(path_value, '/v1/models/') == 1) .and. &
            method%as_character() /= 'POST') then
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
        max_tokens = json_token_limit(body%as_character(), responses, max_tokens_valid)
        if (.not. max_tokens_valid) then
            call error_body(400, 'max_tokens must be a positive integer no greater than 32768', result)
            status = 400_c_int
            call copy_result(result, 'application/json', response, response_capacity, response_length, &
                content_type, content_type_capacity, fortai_native_http_handle); return
        end if
        temperature = json_real(body%as_character(), 'temperature', server_real_default('FORTAI_TEMPERATURE', 0.8_real32), &
            temperature_valid)
        if (.not. temperature_valid .or. .not. finite_real32(temperature) .or. temperature < 0.0_real32) then
            call error_body(400, 'temperature must be a finite non-negative number', result); status = 400_c_int
            call copy_result(result, 'application/json', response, response_capacity, response_length, &
                content_type, content_type_capacity, fortai_native_http_handle); return
        end if
        seed = json_int64(body%as_character(), 'seed', server_int64_default('FORTAI_SEED', 0_int64))
        ! llama.cpp treats -1 as a request for a fresh random seed.  The
        ! native service uses zero for that policy; preserve deterministic
        ! positive seeds while translating the sentinel at the API boundary.
        if (seed < 0_int64) seed = 0_int64
        top_k = fortai_native_http_json_integer_checked(body%as_character(), 'top_k', &
            server_integer_default('FORTAI_TOP_K', 40), top_k_valid)
        repeat_last_n = fortai_native_http_json_integer_checked(body%as_character(), 'repeat_last_n', &
            server_integer_default('FORTAI_REPEAT_LAST_N', 64), repeat_last_n_valid, -1)
        top_p = json_real(body%as_character(), 'top_p', server_real_default('FORTAI_TOP_P', 0.95_real32), top_p_valid)
        min_p = json_real(body%as_character(), 'min_p', server_real_default('FORTAI_MIN_P', 0.05_real32), min_p_valid)
        repeat_penalty = json_real(body%as_character(), 'repeat_penalty', &
            server_real_default('FORTAI_REPEAT_PENALTY', 1.0_real32), repeat_penalty_valid)
        presence_penalty = json_real(body%as_character(), 'presence_penalty', &
            server_real_default('FORTAI_PRESENCE_PENALTY', 0.0_real32), presence_penalty_valid)
        frequency_penalty = json_real(body%as_character(), 'frequency_penalty', &
            server_real_default('FORTAI_FREQUENCY_PENALTY', 0.0_real32), frequency_penalty_valid)
        sampling_valid = top_p_valid .and. min_p_valid .and. repeat_penalty_valid .and. &
            presence_penalty_valid .and. frequency_penalty_valid .and. top_k_valid .and. repeat_last_n_valid
        if (.not. sampling_valid .or. top_k < 0 .or. repeat_last_n < -1 .or. top_p <= 0.0_real32 .or. &
            top_p > 1.0_real32 .or. min_p < 0.0_real32 .or. min_p > 1.0_real32 .or. &
            repeat_penalty <= 0.0_real32) then
            call error_body(400, 'invalid sampling parameters', result); status = 400_c_int
            call copy_result(result, 'application/json', response, response_capacity, response_length, &
                content_type, content_type_capacity, fortai_native_http_handle); return
        end if
        stream = json_boolean(body%as_character(), 'stream', .false.)
        enable_thinking = json_boolean_checked(body%as_character(), 'enable_thinking', &
            fortai_native_service_default_thinking(), enable_thinking_valid, thinking_explicit)
        if (.not. enable_thinking_valid) then
            call error_body(400, 'enable_thinking must be a boolean', result); status = 400_c_int
            call copy_result(result, 'application/json', response, response_capacity, response_length, &
                content_type, content_type_capacity, fortai_native_http_handle); return
        end if
        ! OpenAI-compatible clients use both a top-level field and the
        ! llama.cpp-compatible chat_template_kwargs object.  Responses uses
        ! reasoning.effort; the latter wins when supplied.
        enable_thinking = json_boolean_checked(server_chat_template_kwargs(), 'enable_thinking', enable_thinking, &
            nested_thinking_valid, nested_thinking_found)
        if (.not. nested_thinking_valid) then
            call error_body(500, 'FORTAI_CHAT_TEMPLATE_KWARGS must contain a valid JSON object', result)
            status = 500_c_int
            call copy_result(result, 'application/json', response, response_capacity, response_length, &
                content_type, content_type_capacity, fortai_native_http_handle); return
        end if
        thinking_explicit = thinking_explicit .or. nested_thinking_found
        enable_thinking = json_object_boolean_checked(body%as_character(), 'chat_template_kwargs', &
            'enable_thinking', enable_thinking, nested_thinking_valid, nested_thinking_found)
        if (.not. nested_thinking_valid) then
            call error_body(400, 'chat_template_kwargs.enable_thinking must be a boolean', result)
            status = 400_c_int
            call copy_result(result, 'application/json', response, response_capacity, response_length, &
                content_type, content_type_capacity, fortai_native_http_handle); return
        end if
        thinking_explicit = thinking_explicit .or. nested_thinking_found
        call apply_reasoning_environment(enable_thinking, thinking_explicit)
        reasoning_effort = json_string_checked(body%as_character(), 'reasoning_effort', &
            server_text_default('FORTAI_REASONING_EFFORT', ''), &
            reasoning_effort_type_valid, reasoning_effort_found)
        if (.not. reasoning_effort_type_valid) then
            call error_body(400, 'reasoning_effort must be a string', result); status = 400_c_int
            call copy_result(result, 'application/json', response, response_capacity, response_length, &
                content_type, content_type_capacity, fortai_native_http_handle); return
        end if
        reasoning_effort = json_string_checked(server_chat_template_kwargs(), 'reasoning_effort', reasoning_effort, &
            nested_reasoning_effort_valid, nested_reasoning_effort_found)
        if (.not. nested_reasoning_effort_valid) then
            call error_body(500, 'FORTAI_CHAT_TEMPLATE_KWARGS.reasoning_effort must be a string', result)
            status = 500_c_int
            call copy_result(result, 'application/json', response, response_capacity, response_length, &
                content_type, content_type_capacity, fortai_native_http_handle); return
        end if
        reasoning_effort = json_object_string_checked(body%as_character(), 'chat_template_kwargs', &
            'reasoning_effort', reasoning_effort, nested_reasoning_effort_valid, nested_reasoning_effort_found)
        if (.not. nested_reasoning_effort_valid) then
            call error_body(400, 'chat_template_kwargs.reasoning_effort must be a string', result)
            status = 400_c_int
            call copy_result(result, 'application/json', response, response_capacity, response_length, &
                content_type, content_type_capacity, fortai_native_http_handle); return
        end if
        reasoning_effort = json_object_string_checked(body%as_character(), 'reasoning', 'effort', reasoning_effort, &
            response_reasoning_effort_valid, response_reasoning_effort_found)
        if (.not. response_reasoning_effort_valid) then
            call error_body(400, 'reasoning.effort must be a string', result); status = 400_c_int
            call copy_result(result, 'application/json', response, response_capacity, response_length, &
                content_type, content_type_capacity, fortai_native_http_handle); return
        end if
        reasoning_effort = normalize_reasoning_effort(reasoning_effort, reasoning_effort_valid)
        if (.not. reasoning_effort_valid) then
            call error_body(400, 'reasoning_effort must be one of default, minimal, low, medium, high, xhigh, or max', result)
            status = 400_c_int
            call copy_result(result, 'application/json', response, response_capacity, response_length, &
                content_type, content_type_capacity, fortai_native_http_handle); return
        end if
        select case (trim(reasoning_effort))
        case ('off')
            enable_thinking = .false.
        case ('low', 'medium', 'xhigh')
            if (.not. thinking_explicit) enable_thinking = .true.
        end select
        reasoning_format = json_string_checked(body%as_character(), 'reasoning_format', &
            server_text_default('FORTAI_REASONING_FORMAT', 'auto'), &
            reasoning_format_valid, reasoning_format_found)
        if (.not. reasoning_format_valid) then
            call error_body(400, 'reasoning_format must be a string', result); status = 400_c_int
            call copy_result(result, 'application/json', response, response_capacity, response_length, &
                content_type, content_type_capacity, fortai_native_http_handle); return
        end if
        select case (trim(reasoning_format))
        case ('', 'auto', 'deepseek')
            reasoning_format = 'auto'
        case ('none', 'deepseek-legacy')
            continue
        case default
            call error_body(400, 'reasoning_format must be one of none, auto, deepseek, or deepseek-legacy', result)
            status = 400_c_int
            call copy_result(result, 'application/json', response, response_capacity, response_length, &
                content_type, content_type_capacity, fortai_native_http_handle); return
        end select
        supports_reasoning_effort = fortai_native_service_supports_reasoning_effort()
        supports_preserve_thinking = fortai_native_service_supports_preserve_thinking()
        preserve_thinking = server_boolean_default('FORTAI_REASONING_PRESERVE', supports_preserve_thinking)
        preserve_thinking = json_boolean_checked(server_chat_template_kwargs(), 'preserve_thinking', preserve_thinking, &
            preserve_thinking_valid, preserve_thinking_found)
        if (.not. preserve_thinking_valid) then
            call error_body(500, 'FORTAI_CHAT_TEMPLATE_KWARGS.preserve_thinking must be a boolean', result)
            status = 500_c_int
            call copy_result(result, 'application/json', response, response_capacity, response_length, &
                content_type, content_type_capacity, fortai_native_http_handle); return
        end if
        preserve_thinking = json_boolean_checked(body%as_character(), 'preserve_thinking', preserve_thinking, &
            preserve_thinking_valid, preserve_thinking_found)
        if (.not. preserve_thinking_valid) then
            call error_body(400, 'preserve_thinking must be a boolean', result); status = 400_c_int
            call copy_result(result, 'application/json', response, response_capacity, response_length, &
                content_type, content_type_capacity, fortai_native_http_handle); return
        end if
        preserve_thinking = json_object_boolean_checked(body%as_character(), 'chat_template_kwargs', &
            'preserve_thinking', preserve_thinking, preserve_thinking_valid, preserve_thinking_found)
        if (.not. preserve_thinking_valid) then
            call error_body(400, 'chat_template_kwargs.preserve_thinking must be a boolean', result)
            status = 400_c_int
            call copy_result(result, 'application/json', response, response_capacity, response_length, &
                content_type, content_type_capacity, fortai_native_http_handle); return
        end if
        call reasoning_instruction%clear()
        if (supports_reasoning_effort .and. enable_thinking) then
            reasoning_budget = server_reasoning_budget()
            if (len_trim(reasoning_effort) == 0) reasoning_effort = 'xhigh'
            select case (trim(reasoning_effort))
            case ('xhigh')
                call reasoning_instruction%set('Reasoning effort is set to xhigh. Please think carefully through the task, '&
                    // 'validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, '&
                    // 'and clarity in the final answer.')
            case ('low')
                call reasoning_instruction%set('Reasoning effort is set to low. Keep your thinking brief and focused, moving '&
                    // 'directly to the conclusion without unnecessary elaboration.')
            end select
            if (reasoning_budget > 0) then
                if (reasoning_instruction%length() == 0) then
                    call reasoning_instruction%set('Keep hidden reasoning within a budget of ' // &
                        integer_text(reasoning_budget) // ' tokens.')
                else
                    call reasoning_instruction%append(' Keep hidden reasoning within a budget of ' // &
                        integer_text(reasoning_budget) // ' tokens.')
                end if
            end if
        end if
        call tools_json%clear()
        if (.not. json_array_value(body%as_character(), 'tools', tools_json)) call tools_json%clear()
        if (responses) then
            if (.not. parse_response_messages(body%as_character(), messages, count)) then
                call error_body(400, 'input must contain a string or message array', result); status = 400_c_int
                call copy_result(result, 'application/json', response, response_capacity, response_length, &
                    content_type, content_type_capacity, fortai_native_http_handle); return
            end if
            prompt = format_chat(messages, count, enable_thinking, tools_json, reasoning_instruction, preserve_thinking)
        else if (chat) then
            if (.not. parse_messages(body%as_character(), messages, count)) then
                call error_body(400, 'messages must contain text-only role/content strings; native multimodal ' // &
                    'image/audio input is unsupported', result); status = 400_c_int
                call copy_result(result, 'application/json', response, response_capacity, response_length, &
                    content_type, content_type_capacity, fortai_native_http_handle); return
            end if
            prompt = format_chat(messages, count, enable_thinking, tools_json, reasoning_instruction, preserve_thinking)
        else
            block
                integer :: prompt_value, after
                character(len=7) :: prompt_key
                character(len=:), allocatable :: body_text
                body_text = body%as_character()
                prompt_key = 'prompt'
                prompt_value = json_key(body_text, 'prompt', 1)
                if (prompt_value == 0) then
                    prompt_key = 'content'
                    prompt_value = json_key(body_text, 'content', 1)
                end if
                if (prompt_value == 0 .or. prompt_value > len(body_text)) then
                    call error_body(400, 'prompt must be a string or an array of strings', result)
                    status = 400_c_int
                    call copy_result(result, 'application/json', response, response_capacity, response_length, &
                        content_type, content_type_capacity, fortai_native_http_handle); return
                end if
                if (body_text(prompt_value:prompt_value) == '[') then
                    batch_limit = server_integer_default('FORTAI_BATCH', 2048)
                    if (batch_limit <= 0) batch_limit = 1
                    batch_limit = min(batch_limit, max_messages)
                    if (.not. parse_prompt_array(body_text, trim(prompt_key), prompt_batch, &
                        prompt_batch_count, batch_limit)) then
                        call error_body(400, 'prompt array must contain non-empty strings', result)
                        status = 400_c_int
                        call copy_result(result, 'application/json', response, response_capacity, response_length, &
                            content_type, content_type_capacity, fortai_native_http_handle); return
                    end if
                    prompt_array = .true.
                    if (stream) then
                        call error_body(400, 'streaming prompt arrays are not supported', result)
                        status = 400_c_int
                        call copy_result(result, 'application/json', response, response_capacity, response_length, &
                            content_type, content_type_capacity, fortai_native_http_handle); return
                    end if
                else
                    if (.not. json_string(body_text, prompt_value, prompt, after)) then
                        call error_body(400, 'prompt must be a string or an array of strings', result)
                        status = 400_c_int
                        call copy_result(result, 'application/json', response, response_capacity, response_length, &
                            content_type, content_type_capacity, fortai_native_http_handle); return
                    end if
                    allocate(prompt_batch(1))
                    prompt_batch(1) = prompt
                    prompt_batch_count = 1
                end if
            end block
        end if
        if (chat .or. responses) then
            allocate(prompt_batch(1))
            prompt_batch(1) = prompt
            prompt_batch_count = 1
        end if
        if (prompt_array) then
            allocate(batch_generated(prompt_batch_count), batch_prompt_tokens(prompt_batch_count), &
                batch_tokens(prompt_batch_count))
            do batch_index = 1, prompt_batch_count
                result_code = fortai_native_service_complete_text_sampling(prompt_batch(batch_index)%as_character(), &
                    max_tokens, temperature, seed, top_k, top_p, min_p, repeat_penalty, presence_penalty, &
                    frequency_penalty, repeat_last_n, batch_generated(batch_index), batch_tokens(batch_index), &
                    batch_prompt_tokens(batch_index))
                if (result_code < 0) then
                    call error_body(500, 'FortAI generation failed for prompt batch', result)
                    status = 500_c_int
                    call copy_result(result, 'application/json', response, response_capacity, response_length, &
                        content_type, content_type_capacity, fortai_native_http_handle); return
                end if
            end do
            call completion_batch_body(model_text, batch_generated, batch_prompt_tokens, batch_tokens, result)
            status = 200_c_int
            call copy_result(result, 'application/json', response, response_capacity, response_length, &
                content_type, content_type_capacity, fortai_native_http_handle)
            generation_count = generation_count + int(prompt_batch_count, int64)
            return
        end if
        result_code = fortai_native_service_complete_text_sampling(prompt_batch(1)%as_character(), max_tokens, temperature, &
            seed, top_k, top_p, min_p, repeat_penalty, presence_penalty, frequency_penalty, repeat_last_n, &
            generated, tokens, prompt_tokens)
        if (result_code < 0) then
            call error_body(500, 'FortAI generation failed', result); status = 500_c_int
            call copy_result(result, 'application/json', response, response_capacity, response_length, &
                content_type, content_type_capacity, fortai_native_http_handle); return
        end if
        generation_count = generation_count + 1_int64
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
                result = response_stream_json(model_text, content, reasoning, prompt_tokens, tokens, tool_calls, tool_count)
            else
                result = response_json(model_text, content, reasoning, prompt_tokens, tokens, tool_calls, tool_count)
            end if
        else
            call completion_body(model_text, content, reasoning, prompt_tokens, tokens, chat, stream, result, &
                tool_calls, tool_count)
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

    integer function server_reasoning_budget()
        character(len=32) :: value
        integer :: length, ios, parsed

        server_reasoning_budget = 0
        value = ''
        call get_environment_variable('FORTAI_REASONING_BUDGET', value, length=length)
        if (length <= 0) call get_environment_variable('LLAMACPP_REASONING_BUDGET', value, length=length)
        if (length <= 0 .or. length > len(value)) return
        read(value(:length), *, iostat=ios) parsed
        if (ios == 0 .and. parsed > 0 .and. parsed <= max_generation) server_reasoning_budget = parsed
    end function server_reasoning_budget

    subroutine apply_reasoning_environment(enable_thinking, explicit)
        logical, intent(inout) :: enable_thinking, explicit
        character(len=:), allocatable :: value

        if (explicit) return
        value = server_text_default('FORTAI_REASONING', 'auto')
        select case (trim(value))
        case ('on', 'true', '1')
            enable_thinking = .true.
            explicit = .true.
        case ('off', 'false', '0')
            enable_thinking = .false.
            explicit = .true.
        end select
    end subroutine apply_reasoning_environment

    function integer_text(value) result(text)
        integer, intent(in) :: value
        character(len=32) :: text
        write(text, '(i0)') value
    end function integer_text

    real(real32) function server_real_default(primary, fallback)
        character(len=*), intent(in) :: primary
        real(real32), intent(in) :: fallback
        character(len=64) :: value
        integer :: length, ios

        server_real_default = fallback
        value = ''
        call get_environment_variable(primary, value, length=length)
        if (length <= 0) call get_environment_variable('LLAMACPP_' // primary(8:), value, length=length)
        if (length <= 0 .or. length > len(value)) return
        read(value(:length), *, iostat=ios) server_real_default
        if (ios /= 0) server_real_default = fallback
    end function server_real_default

    logical function server_boolean_default(primary, fallback)
        character(len=*), intent(in) :: primary
        logical, intent(in) :: fallback
        character(len=:), allocatable :: value

        server_boolean_default = fallback
        value = server_text_default(primary, '')
        select case (trim(value))
        case ('1', 'true', 'on', 'yes')
            server_boolean_default = .true.
        case ('0', 'false', 'off', 'no')
            server_boolean_default = .false.
        end select
    end function server_boolean_default

    integer function server_integer_default(primary, fallback)
        character(len=*), intent(in) :: primary
        integer, intent(in) :: fallback
        character(len=64) :: value
        integer :: length, ios

        server_integer_default = fallback
        value = ''
        call get_environment_variable(primary, value, length=length)
        if (length <= 0) call get_environment_variable('LLAMACPP_' // primary(8:), value, length=length)
        if (length <= 0 .or. length > len(value)) return
        read(value(:length), *, iostat=ios) server_integer_default
        if (ios /= 0) server_integer_default = fallback
    end function server_integer_default

    integer(int64) function server_int64_default(primary, fallback)
        character(len=*), intent(in) :: primary
        integer(int64), intent(in) :: fallback
        character(len=64) :: value
        integer :: length, ios

        server_int64_default = fallback
        value = ''
        call get_environment_variable(primary, value, length=length)
        if (length <= 0) call get_environment_variable('LLAMACPP_' // primary(8:), value, length=length)
        if (length <= 0 .or. length > len(value)) return
        read(value(:length), *, iostat=ios) server_int64_default
        if (ios /= 0) server_int64_default = fallback
    end function server_int64_default

    function server_text_default(primary, fallback) result(text)
        character(len=*), intent(in) :: primary, fallback
        character(len=:), allocatable :: text
        character(len=4096) :: value
        integer :: length

        value = ''
        call get_environment_variable(primary, value, length=length)
        if (length <= 0) call get_environment_variable('LLAMACPP_' // primary(8:), value, length=length)
        if (length <= 0 .or. length > len(value)) then
            text = fallback
        else
            text = value(:length)
        end if
    end function server_text_default

    function server_chat_template_kwargs() result(text)
        character(len=:), allocatable :: text
        character(len=4096) :: value
        integer :: length

        value = ''
        call get_environment_variable('FORTAI_CHAT_TEMPLATE_KWARGS', value, length=length)
        if (length <= 0) call get_environment_variable('LLAMA_ARG_CHAT_TEMPLATE_KWARGS', value, length=length)
        if (length <= 0) call get_environment_variable('LLAMACPP_CHAT_TEMPLATE_KWARGS', value, length=length)
        if (length <= 0 .or. length > len(value)) then
            text = '{}'
        else
            text = value(:length)
        end if
    end function server_chat_template_kwargs

    logical function server_web_ui_enabled()
        character(len=16) :: value
        integer :: length

        server_web_ui_enabled = .true.
        value = ''
        call get_environment_variable('FORTAI_NO_WEBUI', value, length=length)
        if (length <= 0) call get_environment_variable('LLAMACPP_NO_WEBUI', value, length=length)
        if (length > 0 .and. length <= len(value)) then
            server_web_ui_enabled = trim(value(:length)) /= '1' .and. trim(value(:length)) /= 'true'
        end if
    end function server_web_ui_enabled

    function server_api_prefix() result(prefix)
        character(len=:), allocatable :: prefix
        character(len=:), allocatable :: configured
        integer :: last

        configured = server_text_default('FORTAI_API_PREFIX', '')
        if (len_trim(configured) == 0) then
            allocate(character(len=0) :: prefix)
            return
        end if
        configured = trim(configured)
        if (configured(1:1) /= '/') configured = '/' // configured
        last = len(configured)
        do while (last > 1)
            if (configured(last:last) /= '/') exit
            last = last - 1
        end do
        if (last < len(configured)) configured = configured(:last)
        allocate(character(len=len(configured)) :: prefix)
        prefix = configured
    end function server_api_prefix

    subroutine apply_api_prefix(path_value)
        character(len=:), allocatable, intent(inout) :: path_value
        character(len=:), allocatable :: prefix

        prefix = server_api_prefix()
        if (len(prefix) == 0) return
        if (path_value == prefix) then
            path_value = '/'
        else if (len(path_value) > len(prefix) .and. index(path_value, prefix // '/') == 1) then
            path_value = path_value(len(prefix) + 1:)
        else
            ! Keep the normal 404 path for requests outside the configured
            ! prefix without allocating a second error response here.
            path_value = '/__fortai_api_prefix_mismatch__'
        end if
    end subroutine apply_api_prefix

    subroutine server_static_file(path_value, result, mime, found)
        character(len=*), intent(in) :: path_value
        type(string_t), intent(out) :: result
        character(len=*), intent(out) :: mime
        logical, intent(out) :: found
        character(len=:), allocatable :: root, relative, file_name, bytes
        integer :: unit, ios
        integer(int64) :: file_size
        logical :: exists

        call result%clear()
        mime = 'application/octet-stream'
        found = .false.
        if (len(path_value) == 0) return
        if (path_value(1:1) /= '/') return
        if (index(path_value, '..') > 0) return
        root = server_text_default('FORTAI_STATIC_PATH', '')
        if (len_trim(root) == 0) return
        relative = path_value
        if (relative == '/') relative = '/index.html'
        root = trim(root)
        if (root(len(root):len(root)) == '/') then
            if (len(relative) > 1) then
                file_name = root // relative(2:)
            else
                file_name = root // 'index.html'
            end if
        else
            file_name = root // relative
        end if
        file_size = 0_int64
        inquire(file=file_name, exist=exists, size=file_size)
        if (.not. exists .or. file_size < 0_int64 .or. file_size > 64_int64 * 1024_int64 * 1024_int64) return
        if (file_size == 0_int64) then
            call result%set('')
        else
            allocate(character(len=int(file_size)) :: bytes)
            open(newunit=unit, file=file_name, status='old', action='read', access='stream', form='unformatted', iostat=ios)
            if (ios /= 0) then
                deallocate(bytes)
                return
            end if
            read(unit, iostat=ios) bytes
            close(unit)
            if (ios /= 0) then
                deallocate(bytes)
                return
            end if
            call result%set(bytes)
            deallocate(bytes)
        end if
        if (index(path_value, '.html') > 0 .or. index(path_value, '.htm') > 0) then
            mime = 'text/html; charset=utf-8'
        else if (index(path_value, '.css') > 0) then
            mime = 'text/css; charset=utf-8'
        else if (index(path_value, '.js') > 0) then
            mime = 'application/javascript; charset=utf-8'
        else if (index(path_value, '.json') > 0) then
            mime = 'application/json'
        else if (index(path_value, '.txt') > 0 .or. index(path_value, '.md') > 0) then
            mime = 'text/plain; charset=utf-8'
        else if (index(path_value, '.svg') > 0) then
            mime = 'image/svg+xml'
        else if (index(path_value, '.png') > 0) then
            mime = 'image/png'
        else if (index(path_value, '.jpg') > 0 .or. index(path_value, '.jpeg') > 0) then
            mime = 'image/jpeg'
        else if (index(path_value, '.ico') > 0) then
            mime = 'image/x-icon'
        end if
        found = .true.
    end subroutine server_static_file

    subroutine server_slots_json(result)
        type(string_t), intent(out) :: result
        character(len=64) :: number
        integer :: slots

        slots = max(1, server_integer_default('FORTAI_PARALLEL', 1))
        call result%set('{"slots":[{"id":0,"state":0,"n_ctx":')
        write(number, '(i0)') fortai_native_service_context_size()
        call result%append(trim(number))
        call result%append(',"is_processing":false,"prompt_tokens":0,"tokens":0}],"total_slots":')
        write(number, '(i0)') slots
        call result%append(trim(number))
        call result%append('}')
    end subroutine server_slots_json

    logical function server_props_update(text, result)
        character(len=*), intent(in) :: text
        type(string_t), intent(out) :: result

        server_props_update = .false.
        call result%clear()
        if (json_top_level_key(text, 'temperature') > 0) then
            if (.not. update_real_property(text, 'temperature', 'FORTAI_TEMPERATURE', 0.0_real32, huge(0.0_real32))) then
                call error_body(400, 'invalid temperature property', result)
                return
            end if
        end if
        if (json_top_level_key(text, 'top_p') > 0) then
            if (.not. update_real_property(text, 'top_p', 'FORTAI_TOP_P', 0.0_real32, 1.0_real32)) then
                call error_body(400, 'invalid top_p property', result)
                return
            end if
        end if
        if (json_top_level_key(text, 'min_p') > 0) then
            if (.not. update_real_property(text, 'min_p', 'FORTAI_MIN_P', 0.0_real32, 1.0_real32)) then
                call error_body(400, 'invalid min_p property', result)
                return
            end if
        end if
        if (json_top_level_key(text, 'repeat_penalty') > 0) then
            if (.not. update_real_property(text, 'repeat_penalty', 'FORTAI_REPEAT_PENALTY', tiny(1.0_real32), &
                huge(0.0_real32))) then
                call error_body(400, 'invalid repeat_penalty property', result)
                return
            end if
        end if
        if (json_top_level_key(text, 'presence_penalty') > 0) then
            if (.not. update_real_property(text, 'presence_penalty', 'FORTAI_PRESENCE_PENALTY', -huge(0.0_real32), &
                huge(0.0_real32))) then
                call error_body(400, 'invalid presence_penalty property', result)
                return
            end if
        end if
        if (json_top_level_key(text, 'frequency_penalty') > 0) then
            if (.not. update_real_property(text, 'frequency_penalty', 'FORTAI_FREQUENCY_PENALTY', -huge(0.0_real32), &
                huge(0.0_real32))) then
                call error_body(400, 'invalid frequency_penalty property', result)
                return
            end if
        end if
        if (json_top_level_key(text, 'top_k') > 0) then
            if (.not. update_integer_property(text, 'top_k', 'FORTAI_TOP_K', 0, max_generation)) then
                call error_body(400, 'invalid top_k property', result)
                return
            end if
        end if
        if (json_top_level_key(text, 'repeat_last_n') > 0) then
            if (.not. update_integer_property(text, 'repeat_last_n', 'FORTAI_REPEAT_LAST_N', -1, max_generation)) then
                call error_body(400, 'invalid repeat_last_n property', result)
                return
            end if
        end if
        server_props_update = .true.
    end function server_props_update

    logical function update_real_property(text, key, environment, minimum, maximum)
        character(len=*), intent(in) :: text, key, environment
        real(real32), intent(in) :: minimum, maximum
        real(real32) :: value
        logical :: valid
        character(len=64) :: encoded

        value = json_real(text, key, minimum, valid)
        update_real_property = valid .and. finite_real32(value) .and. value >= minimum .and. value <= maximum
        if (.not. update_real_property) return
        write(encoded, '(es24.16)') value
        update_real_property = set_http_environment(environment, trim(encoded))
    end function update_real_property

    logical function update_integer_property(text, key, environment, minimum, maximum)
        character(len=*), intent(in) :: text, key, environment
        integer, intent(in) :: minimum, maximum
        integer :: value
        logical :: valid
        character(len=64) :: encoded

        value = fortai_native_http_json_integer_checked(text, key, minimum, valid, minimum)
        update_integer_property = valid .and. value >= minimum .and. value <= maximum
        if (.not. update_integer_property) return
        write(encoded, '(i0)') value
        update_integer_property = set_http_environment(environment, trim(encoded))
    end function update_integer_property

    logical function set_http_environment(name, value)
        character(len=*), intent(in) :: name, value
        type(string_t) :: name_text, value_text
        character(kind=c_char), allocatable :: cname(:), cvalue(:)
        integer(c_int) :: status

        call name_text%set(name)
        call value_text%set(value)
        allocate(cname(name_text%length() + 1), cvalue(value_text%length() + 1))
        call name_text%to_c(cname, size(cname))
        call value_text%to_c(cvalue, size(cvalue))
        status = fortai_server_set_environment_http(cname, cvalue)
        set_http_environment = status == 0_c_int
    end function set_http_environment

    subroutine metrics_body(result, cuda)
        type(string_t), intent(out) :: result
        integer(c_int), intent(in) :: cuda
        character(len=64) :: number

        call result%set('# HELP fortai_requests_total HTTP requests handled by FortAI.' // char(10) // &
            '# TYPE fortai_requests_total counter' // char(10) // 'fortai_requests_total ')
        write(number, '(i0)') request_count
        call result%append(trim(number) // char(10) // '# HELP fortai_generation_requests_total completed generations.' // &
            char(10) // '# TYPE fortai_generation_requests_total counter' // char(10) // &
            'fortai_generation_requests_total ')
        write(number, '(i0)') generation_count
        call result%append(trim(number) // char(10) // 'fortai_cuda_enabled ')
        call result%append(merge('1', '0', cuda /= 0_c_int) // char(10))
    end subroutine metrics_body

    subroutine server_model_id(model)
        type(string_t), intent(inout) :: model
        character(len=256) :: alias
        integer :: alias_length

        alias = ''
        call get_environment_variable('FORTAI_SERVER_ALIAS', alias, length=alias_length)
        if (alias_length > 0 .and. alias_length <= len(alias)) call model%set(alias(:alias_length))
    end subroutine server_model_id

    subroutine server_settings_json(result)
        type(string_t), intent(inout) :: result
        character(len=64) :: number

        call result%append('{')
        call append_setting(result, 'alias', 'FORTAI_SERVER_ALIAS', 'qwen', .true.)
        call result%append(',"context":')
        write(number, '(i0)') fortai_native_service_context_size()
        call result%append(trim(number))
        call append_setting(result, 'threads', 'FORTAI_THREADS', '0', .false.)
        call append_setting(result, 'gpu_layers', 'FORTAI_GPU_LAYERS', '0', .false.)
        call append_setting(result, 'main_gpu', 'FORTAI_MAIN_GPU', '0', .false.)
        call append_setting(result, 'parallel', 'FORTAI_PARALLEL', '1', .false.)
        call append_setting(result, 'slot_mode', 'FORTAI_SLOT_MODE', 'serialized', .true.)
        call append_setting(result, 'batch', 'FORTAI_BATCH', '2048', .false.)
        call append_setting(result, 'ubatch', 'FORTAI_UBATCH', '512', .false.)
        call append_setting(result, 'tensor_split', 'FORTAI_TENSOR_SPLIT', '', .true.)
        call append_setting(result, 'split_mode', 'FORTAI_SPLIT_MODE', 'layer', .true.)
        call append_setting(result, 'flash_attn', 'FORTAI_FLASH_ATTN', 'auto', .true.)
        call append_setting(result, 'cache_type_k', 'FORTAI_CACHE_TYPE_K', 'f16', .true.)
        call append_setting(result, 'cache_type_v', 'FORTAI_CACHE_TYPE_V', 'f16', .true.)
        call append_setting(result, 'cache_type_k_draft', 'FORTAI_CACHE_TYPE_K_DRAFT', 'f16', .true.)
        call append_setting(result, 'cache_type_v_draft', 'FORTAI_CACHE_TYPE_V_DRAFT', 'f16', .true.)
        call append_setting(result, 'draft_model', 'FORTAI_DRAFT_MODEL', '', .true.)
        call append_setting(result, 'spec_type', 'FORTAI_SPEC_TYPE', '', .true.)
        call append_setting(result, 'spec_draft_n_max', 'FORTAI_SPEC_DRAFT_N_MAX', '0', .false.)
        call append_setting(result, 'reasoning_budget', 'FORTAI_REASONING_BUDGET', '0', .false.)
        call append_setting(result, 'reasoning_effort', 'FORTAI_REASONING_EFFORT', 'default', .true.)
        call append_setting(result, 'reasoning', 'FORTAI_REASONING', 'auto', .true.)
        call append_setting(result, 'reasoning_preserve', 'FORTAI_REASONING_PRESERVE', 'false', .false.)
        call append_setting(result, 'chat_template_kwargs', 'FORTAI_CHAT_TEMPLATE_KWARGS', '{}', .true.)
        call append_setting(result, 'mmproj', 'FORTAI_MMPROJ', '', .true.)
        call append_setting(result, 'mmproj_offload', 'FORTAI_MMPROJ_OFFLOAD', 'true', .false.)
        call append_setting(result, 'mmproj_device', 'MTMD_BACKEND_DEVICE', 'auto', .true.)
        call append_setting(result, 'load_mode', 'FORTAI_LOAD_MODE', 'auto', .true.)
        call append_setting(result, 'fit', 'FORTAI_FIT', 'auto', .true.)
        call append_setting(result, 'cache_ram', 'FORTAI_CACHE_RAM', '0', .false.)
        call append_setting(result, 'cache_reuse', 'FORTAI_CACHE_REUSE', '0', .false.)
        call result%append(',"cache_reuse_supported":')
        if (fortai_native_service_cache_reuse_supported()) then
            call result%append('true')
        else
            call result%append('false')
        end if
        call result%append(',"cache_reuse_active":')
        if (fortai_native_service_cache_reuse_active()) then
            call result%append('true')
        else
            call result%append('false')
        end if
        call result%append(',"cache_reuse_tokens":')
        write(number, '(i0)') fortai_native_service_cache_reuse_count()
        call result%append(trim(number))
        call append_setting(result, 'n_cpu_moe', 'FORTAI_N_CPU_MOE', '0', .false.)
        call append_setting(result, 'threads_http', 'FORTAI_THREADS_HTTP', '0', .false.)
        call append_setting(result, 'no_context_shift', 'FORTAI_NO_CONTEXT_SHIFT', 'false', .false.)
        call append_setting(result, 'metrics', 'FORTAI_METRICS', 'false', .false.)
        call append_setting(result, 'log_timestamps', 'FORTAI_LOG_TIMESTAMPS', 'false', .false.)
        call append_setting(result, 'temperature', 'FORTAI_TEMPERATURE', '0.8', .false.)
        call append_setting(result, 'top_k', 'FORTAI_TOP_K', '40', .false.)
        call append_setting(result, 'top_p', 'FORTAI_TOP_P', '0.95', .false.)
        call append_setting(result, 'min_p', 'FORTAI_MIN_P', '0.05', .false.)
        call append_setting(result, 'repeat_penalty', 'FORTAI_REPEAT_PENALTY', '1', .false.)
        call append_setting(result, 'presence_penalty', 'FORTAI_PRESENCE_PENALTY', '0', .false.)
        call append_setting(result, 'frequency_penalty', 'FORTAI_FREQUENCY_PENALTY', '0', .false.)
        call append_setting(result, 'repeat_last_n', 'FORTAI_REPEAT_LAST_N', '64', .false.)
        call append_setting(result, 'reasoning_format', 'FORTAI_REASONING_FORMAT', 'auto', .true.)
        call append_setting(result, 'static_path', 'FORTAI_STATIC_PATH', '', .true.)
        call append_setting(result, 'api_prefix', 'FORTAI_API_PREFIX', '', .true.)
        call append_setting(result, 'cors_origins', 'FORTAI_CORS_ORIGINS', '*', .true.)
        call append_setting(result, 'cors_methods', 'FORTAI_CORS_METHODS', 'GET, POST, DELETE, OPTIONS', .true.)
        call append_setting(result, 'cors_headers', 'FORTAI_CORS_HEADERS', '*', .true.)
        call append_setting(result, 'cors_credentials', 'FORTAI_CORS_CREDENTIALS', 'true', .false.)
        call append_setting(result, 'tools', 'FORTAI_TOOLS', '', .true.)
        call append_setting(result, 'tools_runtime', 'FORTAI_TOOLS_RUNTIME', '', .true.)
        call append_setting(result, 'mcp_servers_config', 'FORTAI_MCP_SERVERS_CONFIG', '', .true.)
        call append_setting(result, 'mcp_servers_json', 'FORTAI_MCP_SERVERS_JSON', '', .true.)
        call append_setting(result, 'embedding', 'FORTAI_EMBEDDINGS', 'false', .false.)
        call append_setting(result, 'reranking', 'FORTAI_RERANKING', 'false', .false.)
        call append_setting(result, 'endpoint_props', 'FORTAI_ENDPOINT_PROPS', 'false', .false.)
        call append_setting(result, 'endpoint_slots', 'FORTAI_ENDPOINT_SLOTS', 'true', .false.)
        call append_setting(result, 'no_webui', 'FORTAI_NO_WEBUI', 'false', .false.)
        call result%append('}')
    end subroutine server_settings_json

    subroutine server_props_json(model, model_path, result)
        type(string_t), intent(in) :: model, model_path
        type(string_t), intent(out) :: result
        character(len=64) :: number

        call result%set('{"default_generation_settings":{"params":')
        call server_settings_json(result)
        call result%append(',"n_ctx":')
        write(number, '(i0)') fortai_native_service_context_size()
        call result%append(trim(number))
        call result%append('},"total_slots":')
        write(number, '(i0)') server_integer_default('FORTAI_PARALLEL', 1)
        call result%append(trim(number))
        call result%append(',"model_alias":')
        result = append_json(result, model)
        call result%append(',"model_path":')
        result = append_json(result, model_path)
        call result%append(',"modalities":{"vision":false,"video":false,"audio":false}')
        call result%append(',"endpoint_slots":')
        call result%append_logical(server_boolean_default('FORTAI_ENDPOINT_SLOTS', .true.))
        call result%append(',"endpoint_props":')
        call result%append_logical(server_boolean_default('FORTAI_ENDPOINT_PROPS', .false.))
        call result%append(',"endpoint_metrics":')
        call result%append_logical(server_boolean_default('FORTAI_METRICS', .false.))
        call result%append(',"ui":')
        call result%append_logical(server_web_ui_enabled())
        call result%append(',"build_info":"fortai-native","is_sleeping":false}')
    end subroutine server_props_json

    subroutine append_setting(result, key, environment, fallback, quoted)
        type(string_t), intent(inout) :: result
        character(len=*), intent(in) :: key, environment, fallback
        logical, intent(in) :: quoted
        character(len=4096) :: value
        integer :: length
        type(string_t) :: escaped, raw
        character(len=:), allocatable :: existing, scalar

        value = ''
        call get_environment_variable(environment, value, length=length)
        if (length <= 0) then
            value = fallback
            length = len_trim(fallback)
        else if (length > len(value)) then
            value = fallback
            length = len_trim(fallback)
        end if
        existing = result%as_character()
        if (len(existing) > 0) then
            if (existing(len(existing):len(existing)) /= '{') call result%append(',')
        end if
        call result%append('"' // key // '":')
        if (quoted) then
            if (length > 0) then
                call raw%set(value(:length))
                escaped = json_escape(raw)
            else
                call escaped%set('""')
            end if
            call result%append_string(escaped)
        else
            if (length <= 0) then
                scalar = '0'
            else
                scalar = trim(value(:length))
                if (len(scalar) > 0) then
                    if (scalar(1:1) == '.') then
                        scalar = '0' // scalar
                    else if (len(scalar) >= 2) then
                        if (scalar(1:2) == '-.') scalar = '-0' // scalar(2:)
                    end if
                else
                    scalar = '0'
                end if
            end if
            call result%append(scalar)
        end if
    end subroutine append_setting

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

    logical function is_whisper_model_path(path)
        character(len=*), intent(in) :: path
        character(len=:), allocatable :: lowered
        integer :: n

        lowered = path
        do n = 1, len(lowered)
            if (iachar(lowered(n:n)) >= iachar('A') .and. iachar(lowered(n:n)) <= iachar('Z')) then
                lowered(n:n) = achar(iachar(lowered(n:n)) + 32)
            end if
        end do
        is_whisper_model_path = ends_with(lowered, '.bin') .or. ends_with(lowered, '.ggml')
    contains
        logical function ends_with(value, suffix)
            character(len=*), intent(in) :: value, suffix
            integer :: value_length, suffix_length

            value_length = len_trim(value)
            suffix_length = len(suffix)
            ends_with = value_length >= suffix_length
            if (ends_with) ends_with = value(value_length - suffix_length + 1:value_length) == suffix
        end function ends_with
    end function is_whisper_model_path

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
