module fortai_whisper_service
    !! Native Whisper HTTP-facing service state.
    !!
    !! The transport passes bounded HTTP bytes through ISO C.  This module
    !! owns all route policy, WAV extraction, inference, token decoding, and
    !! JSON escaping; no whisper.cpp/libwhisper call is made here.
    use, intrinsic :: iso_c_binding, only: c_char, c_int, c_null_char
    use, intrinsic :: iso_fortran_env, only: int32, int64, real32
    use fortai_string, only: string_t
    use fortai_status, only: FORTAI_INVALID, FORTAI_UNSUPPORTED, status_t
    use fortai_whisper_audio_io, only: whisper_wav_decode
    use fortai_whisper_runtime, only: whisper_native_runtime_t
    implicit none
    private

    type(whisper_native_runtime_t), save :: service_runtime
    logical, save :: service_ready = .false.
    character(len=:), allocatable, save :: service_model_path

    public :: fortai_whisper_service_init
    public :: fortai_whisper_service_close
    public :: fortai_whisper_service_ready
    public :: fortai_whisper_service_memory_bytes
    public :: fortai_whisper_http_handle

contains

    subroutine fortai_whisper_service_init(path, use_gpu, gpu_device, flash_attention, threads, stat)
        character(len=*), intent(in) :: path
        logical, intent(in), optional :: use_gpu, flash_attention
        integer(int32), intent(in), optional :: gpu_device, threads
        type(status_t), intent(out) :: stat

        call stat%clear()
        call fortai_whisper_service_close()
        call service_runtime%init(trim(path), use_gpu, gpu_device, flash_attention, threads, stat)
        if (.not. stat%is_ok()) return
        service_model_path = trim(path)
        service_ready = .true.
    end subroutine fortai_whisper_service_init

    subroutine fortai_whisper_service_close()
        call service_runtime%close()
        service_ready = .false.
        if (allocated(service_model_path)) deallocate(service_model_path)
    end subroutine fortai_whisper_service_close

    logical function fortai_whisper_service_ready()
        fortai_whisper_service_ready = .false.
        if (.not. service_ready) return
        fortai_whisper_service_ready = service_runtime%is_ready()
    end function fortai_whisper_service_ready

    integer(int64) function fortai_whisper_service_memory_bytes()
        fortai_whisper_service_memory_bytes = 0_int64
        if (.not. fortai_whisper_service_ready()) return
        fortai_whisper_service_memory_bytes = service_runtime%memory_bytes()
    end function fortai_whisper_service_memory_bytes

    integer(c_int) function fortai_whisper_http_handle(request, request_length, model, cuda, response, &
            response_capacity, response_length, status, content_type, content_type_capacity) &
            bind(C, name='fortai_whisper_http_handle')
        character(kind=c_char), intent(in) :: request(*), model(*)
        integer(c_int), value, intent(in) :: request_length, cuda, response_capacity, content_type_capacity
        character(kind=c_char), intent(out) :: response(*), content_type(*)
        integer(c_int), intent(out) :: response_length, status
        type(string_t) :: raw_text, model_text, method, path, body, mime, result, text
        type(status_t) :: stat
        character(len=:), allocatable :: raw, body_value, path_value, method_value, task, language, field_value
        logical :: okay, found, field_found, is_audio, is_translation, is_multipart
        integer(int32) :: max_tokens
        real(real32) :: temperature
        integer(int64) :: seed

        response_length = 0_c_int
        status = 500_c_int
        call raw_text%from_c(request, int(max(0_c_int, request_length)))
        call model_text%from_c(model)
        raw = raw_text%as_character()
        call parse_whisper_request(raw, method, path, body, mime, okay)
        if (.not. okay) then
            call result%set(whisper_error_json('malformed HTTP request'))
            status = 400_c_int
            fortai_whisper_http_handle = whisper_copy_result(result, 'application/json', response, response_capacity, &
                response_length, content_type, content_type_capacity)
            return
        end if
        method_value = method%as_character()
        path_value = path%as_character()
        body_value = body%as_character()
        is_multipart = index(whisper_lower_ascii(mime%as_character()), 'multipart/form-data') == 1
        if (path_value == '/' .or. path_value == '/ui' .or. path_value == '/index.html') then
            if (method_value /= 'GET') then
                call result%set(whisper_error_json('method not allowed'))
                status = 400_c_int
                fortai_whisper_http_handle = whisper_copy_result(result, 'application/json', response, response_capacity, &
                    response_length, content_type, content_type_capacity)
            else
                call result%set(whisper_ui())
                status = 200_c_int
                fortai_whisper_http_handle = whisper_copy_result(result, 'text/html; charset=utf-8', response, response_capacity, &
                    response_length, content_type, content_type_capacity)
            end if
            return
        end if
        if (path_value == '/health' .or. path_value == '/v1/health') then
            call result%set(whisper_health_json(model_text%as_character(), cuda))
            status = merge(200_c_int, 503_c_int, fortai_whisper_service_ready())
            fortai_whisper_http_handle = whisper_copy_result(result, 'application/json', response, response_capacity, &
                response_length, content_type, content_type_capacity)
            return
        end if
        if (path_value == '/models' .or. path_value == '/v1/models') then
            call result%set(whisper_models_json(model_text%as_character()))
            status = 200_c_int
            fortai_whisper_http_handle = whisper_copy_result(result, 'application/json', response, response_capacity, &
                response_length, content_type, content_type_capacity)
            return
        end if
        if (path_value == '/metrics') then
            if (fortai_whisper_service_ready()) then
                call result%set('fortai_whisper_ready 1' // new_line('a'))
            else
                call result%set('fortai_whisper_ready 0' // new_line('a'))
            end if
            call result%append('fortai_whisper_memory_bytes ')
            call result%append(int64_text(fortai_whisper_service_memory_bytes()))
            call result%append(new_line('a'))
            status = 200_c_int
            fortai_whisper_http_handle = whisper_copy_result(result, 'text/plain; version=0.0.4', response, response_capacity, &
                response_length, content_type, content_type_capacity)
            return
        end if
        is_translation = path_value == '/v1/audio/translations' .or. path_value == '/audio/translations'
        is_audio = is_translation .or. path_value == '/v1/audio/transcriptions' .or. &
            path_value == '/audio/transcriptions' .or. path_value == '/inference'
        if (.not. is_audio .or. method_value /= 'POST') then
            call result%set(whisper_error_json('Whisper endpoint not found'))
            status = 404_c_int
            fortai_whisper_http_handle = whisper_copy_result(result, 'application/json', response, response_capacity, &
                response_length, content_type, content_type_capacity)
            return
        end if
        if (.not. fortai_whisper_service_ready()) then
            call result%set(whisper_error_json('Whisper model is not ready'))
            status = 503_c_int
            fortai_whisper_http_handle = whisper_copy_result(result, 'application/json', response, response_capacity, &
                response_length, content_type, content_type_capacity)
            return
        end if
        max_tokens = whisper_json_int(body_value, 'max_tokens', 256_int32, found)
        if (is_multipart) then
            call whisper_form_value(body_value, mime%as_character(), 'max_tokens', field_value, field_found)
            if (field_found) max_tokens = whisper_text_int(field_value, max_tokens, found)
        end if
        if (.not. found) max_tokens = 256_int32
        if (max_tokens <= 0 .or. max_tokens > 4096) then
            call result%set(whisper_error_json('max_tokens must be between 1 and 4096'))
            status = 400_c_int
            fortai_whisper_http_handle = whisper_copy_result(result, 'application/json', response, response_capacity, &
                response_length, content_type, content_type_capacity)
            return
        end if
        temperature = whisper_json_real(body_value, 'temperature', 0.0_real32, found)
        if (is_multipart) then
            call whisper_form_value(body_value, mime%as_character(), 'temperature', field_value, field_found)
            if (field_found) temperature = whisper_text_real(field_value, temperature, found)
        end if
        if (.not. found) temperature = 0.0_real32
        if (.not. finite_real32(temperature) .or. temperature < 0.0_real32) then
            call result%set(whisper_error_json('temperature must be finite and non-negative'))
            status = 400_c_int
            fortai_whisper_http_handle = whisper_copy_result(result, 'application/json', response, response_capacity, &
                response_length, content_type, content_type_capacity)
            return
        end if
        seed = whisper_json_int64(body_value, 'seed', 0_int64, found)
        if (is_multipart) then
            call whisper_form_value(body_value, mime%as_character(), 'seed', field_value, field_found)
            if (field_found) seed = whisper_text_int64(field_value, seed, found)
        end if
        if (.not. found) seed = 0_int64
        language = whisper_json_string(body_value, 'language', 'en', found)
        if (is_multipart) then
            call whisper_form_value(body_value, mime%as_character(), 'language', field_value, field_found)
            if (field_found) language = trim(field_value)
        end if
        task = 'transcribe'
        if (is_translation) task = 'translate'
        if (.not. is_translation) then
            task = whisper_json_string(body_value, 'task', 'transcribe', found)
            if (is_multipart) then
                call whisper_form_value(body_value, mime%as_character(), 'task', field_value, field_found)
                if (field_found) task = trim(field_value)
            end if
        end if
        call whisper_extract_audio(body_value, mime%as_character(), text, is_audio, stat)
        if (.not. stat%is_ok()) then
            call result%set(whisper_error_json(stat%message))
            status = 400_c_int
            fortai_whisper_http_handle = whisper_copy_result(result, 'application/json', response, response_capacity, &
                response_length, content_type, content_type_capacity)
            return
        end if
        call whisper_transcribe_payload(text%as_character(), task, max_tokens, temperature, seed, result, stat, language)
        if (.not. stat%is_ok()) then
            call result%set(whisper_error_json(stat%message))
            if (stat%code == FORTAI_INVALID .or. stat%code == FORTAI_UNSUPPORTED) then
                status = 400_c_int
            else
                status = 500_c_int
            end if
        else
            status = 200_c_int
        end if
        fortai_whisper_http_handle = whisper_copy_result(result, 'application/json', response, response_capacity, &
            response_length, content_type, content_type_capacity)
    end function fortai_whisper_http_handle

    subroutine parse_whisper_request(raw, method, path, body, mime, okay)
        character(len=*), intent(in) :: raw
        type(string_t), intent(out) :: method, path, body, mime
        logical, intent(out) :: okay
        integer :: first_space, second_space, header_end, content_position, line_end, value_start, value_end
        character(len=:), allocatable :: header, lowered

        call method%clear(); call path%clear(); call body%clear(); call mime%clear(); okay = .false.
        first_space = index(raw, ' ')
        if (first_space <= 1) return
        second_space = index(raw(first_space + 1:), ' ')
        if (second_space <= 1) return
        second_space = first_space + second_space
        call method%set(raw(:first_space - 1))
        value_start = first_space + 1
        value_end = second_space - 1
        content_position = index(raw(value_start:value_end), '?')
        if (content_position > 0) value_end = value_start + content_position - 2
        if (value_end >= value_start) call path%set(raw(value_start:value_end))
        header_end = index(raw, char(13) // char(10) // char(13) // char(10))
        if (header_end <= 0) return
        header = raw(:header_end - 1)
        call body%set(raw(header_end + 4:))
        lowered = whisper_lower_ascii(header)
        line_end = index(lowered, 'content-type:')
        if (line_end > 0) then
            value_start = line_end + len('content-type:')
            do while (value_start <= len(header))
                if (header(value_start:value_start) /= ' ' .and. header(value_start:value_start) /= char(9)) exit
                value_start = value_start + 1
            end do
            value_end = index(header(value_start:), char(13) // char(10))
            if (value_end > 0) then
                value_end = value_start + value_end - 2
            else
                value_end = len(header)
            end if
            if (value_end >= value_start) call mime%set(header(value_start:value_end))
        end if
        okay = method%length() > 0 .and. path%length() > 0
    end subroutine parse_whisper_request

    subroutine whisper_extract_audio(raw_body, content_type, audio, is_audio, stat)
        character(len=*), intent(in) :: raw_body, content_type
        type(string_t), intent(out) :: audio
        logical, intent(in) :: is_audio
        type(status_t), intent(out) :: stat
        character(len=:), allocatable :: lower_type, boundary, marker, part_header
        integer :: boundary_pos, start, stop, header_end, marker_pos, search_start, relative_pos
        integer :: payload_start, payload_end
        logical :: file_part

        call audio%clear(); call stat%clear()
        if (.not. is_audio) then
            call stat%set(FORTAI_INVALID, 'audio endpoint is invalid')
            return
        end if
        lower_type = whisper_lower_ascii(content_type)
        if (index(lower_type, 'multipart/form-data') == 1) then
            boundary_pos = index(lower_type, 'boundary=')
            if (boundary_pos <= 0) then
                call stat%set(FORTAI_INVALID, 'multipart audio is missing its boundary')
                return
            end if
            start = boundary_pos + len('boundary=')
            stop = len_trim(content_type)
            if (start <= stop) then
                if (content_type(start:start) == '"') then
                    start = start + 1
                    if (start <= stop) then
                        if (content_type(stop:stop) == '"') stop = stop - 1
                    end if
                end if
            end if
            if (stop < start) then
                call stat%set(FORTAI_INVALID, 'multipart audio boundary is empty')
                return
            end if
            boundary = content_type(start:stop)
            marker = '--' // boundary
            search_start = 1
            do while (search_start <= len(raw_body))
                relative_pos = index(raw_body(search_start:), marker)
                if (relative_pos <= 0) exit
                marker_pos = search_start + relative_pos - 1
                header_end = index(raw_body(marker_pos:), char(13) // char(10) // char(13) // char(10))
                if (header_end <= 0) exit
                payload_start = marker_pos + header_end + 3
                if (payload_start > len(raw_body)) exit
                payload_end = index(raw_body(payload_start:), char(13) // char(10) // marker)
                if (payload_end <= 0) exit
                payload_end = payload_start + payload_end - 2
                part_header = whisper_lower_ascii(raw_body(marker_pos:payload_start - 1))
                file_part = index(part_header, 'filename=') > 0 .or. index(part_header, 'name="file"') > 0 .or. &
                    index(part_header, 'name="audio"') > 0 .or. index(part_header, 'content-type: audio/') > 0
                if (file_part) then
                    if (payload_end >= payload_start) call audio%set(raw_body(payload_start:payload_end))
                    exit
                end if
                search_start = payload_end + 2
            end do
            if (audio%length() == 0) call stat%set(FORTAI_INVALID, 'multipart audio file part is missing or truncated')
        else
            call audio%set(raw_body)
        end if
        if (audio%length() < 12) call stat%set(FORTAI_INVALID, 'audio payload is empty or truncated')
    end subroutine whisper_extract_audio

    subroutine whisper_transcribe_payload(raw, task, max_tokens, temperature, seed, result, stat, language)
        character(len=*), intent(in) :: raw, task
        integer(int32), intent(in) :: max_tokens
        real(real32), intent(in) :: temperature
        integer(int64), intent(in) :: seed
        type(string_t), intent(out) :: result
        type(status_t), intent(out) :: stat
        character(len=*), intent(in), optional :: language
        real(real32), allocatable :: samples(:)
        type(string_t) :: text
        character(len=:), allocatable :: raw_text

        call result%clear(); call stat%clear()
        raw_text = raw
        call whisper_wav_to_samples(raw_text, samples, stat)
        if (.not. stat%is_ok()) return
        if (present(language)) then
            call service_runtime%transcribe(samples, text, stat, language, task, max_tokens, temperature, seed)
        else
            call service_runtime%transcribe(samples, text, stat, 'en', task, max_tokens, temperature, seed)
        end if
        deallocate(samples)
        if (.not. stat%is_ok()) return
        call result%set(whisper_transcription_json(text%as_character()))
    end subroutine whisper_transcribe_payload

    subroutine whisper_wav_to_samples(raw, samples, stat)
        character(len=*), intent(in) :: raw
        real(real32), allocatable, intent(out) :: samples(:)
        type(status_t), intent(out) :: stat
        call whisper_wav_decode(raw, samples, stat)
    end subroutine whisper_wav_to_samples

    function whisper_transcription_json(text) result(result)
        character(len=*), intent(in) :: text
        character(len=:), allocatable :: result
        type(string_t) :: escaped

        escaped = whisper_json_escape(text)
        result = '{"text":"' // escaped%as_character() // '"}'
    end function whisper_transcription_json

    function whisper_error_json(message) result(result)
        character(len=*), intent(in) :: message
        character(len=:), allocatable :: result
        type(string_t) :: escaped

        escaped = whisper_json_escape(message)
        result = '{"error":{"message":"' // escaped%as_character() // '","type":"invalid_request_error"}}'
    end function whisper_error_json

    function whisper_health_json(model, cuda) result(result)
        character(len=*), intent(in) :: model
        integer(c_int), intent(in) :: cuda
        character(len=:), allocatable :: result

        type(string_t) :: escaped
        character(len=7) :: state_text
        character(len=5) :: cuda_text
        character(len=32) :: memory_text
        integer(int64) :: memory_bytes

        if (fortai_whisper_service_ready()) then
            state_text = 'ok'
        else
            state_text = 'loading'
        end if
        if (cuda /= 0) then
            cuda_text = 'true'
        else
            cuda_text = 'false'
        end if
        escaped = whisper_json_escape(model)
        memory_bytes = fortai_whisper_service_memory_bytes()
        write(memory_text, '(i0)') memory_bytes
        result = '{"status":"' // trim(state_text) // &
            '","backend":"fortai-whisper","cuda":' // trim(cuda_text) // &
            ',"model":"' // escaped%as_character() // '","memory_bytes":' // trim(memory_text) // '}'
    end function whisper_health_json

    function whisper_models_json(model) result(result)
        character(len=*), intent(in) :: model
        character(len=:), allocatable :: result
        type(string_t) :: escaped

        escaped = whisper_json_escape(model)
        result = '{"object":"list","data":[{"id":"' // escaped%as_character() // &
            '","object":"model","owned_by":"fortai","permission":[]}]}'
    end function whisper_models_json

    function whisper_ui() result(page)
        character(len=:), allocatable :: page

        page = '<!doctype html><html><head><meta charset="utf-8"><title>FortAI Whisper</title>' // &
            '<style>body{font:16px system-ui;background:#10131a;color:#edf2f7;max-width:760px;margin:4rem auto;padding:0 1rem}' // &
            'button{padding:.6rem 1rem;background:#5b8cff;color:white;border:0;border-radius:.4rem}' // &
            'pre{white-space:pre-wrap;background:#1b2130;padding:1rem;border-radius:.4rem}' // &
            'input{margin:.8rem 0}</style></head><body><h1>FortAI Whisper</h1>' // &
            '<p>Native large-v3-turbo transcription</p><input id="audio" type="file" accept="audio/*">' // &
            '<button id="go">Transcribe</button><pre id="out"></pre><script>' // &
            'go.onclick=async()=>{const f=audio.files[0];if(!f)return;out.textContent="working...";' // &
            'const b=new FormData();b.append("file",f);const r=await fetch("/v1/audio/transcriptions",{method:"POST",body:b});' // &
            'out.textContent=JSON.stringify(await r.json(),null,2)}</script></body></html>'
    end function whisper_ui

    function int64_text(value) result(text)
        integer(int64), intent(in) :: value
        character(len=:), allocatable :: text
        character(len=32) :: buffer

        write(buffer, '(i0)') value
        text = trim(buffer)
    end function int64_text

    integer(c_int) function whisper_copy_result(result, mime, response, capacity, response_length, content_type, mime_capacity)
        type(string_t), intent(in) :: result
        character(len=*), intent(in) :: mime
        character(kind=c_char), intent(out) :: response(*), content_type(*)
        integer(c_int), value, intent(in) :: capacity, mime_capacity
        integer(c_int), intent(out) :: response_length
        integer :: i, required, mime_length
        character(len=:), allocatable :: value

        required = result%length()
        response_length = 0_c_int
        if (capacity <= required) then
            whisper_copy_result = -int(required, c_int)
            return
        end if
        value = result%as_character()
        do i = 1, required
            response(i) = value(i:i)
        end do
        response(required + 1) = c_null_char
        response_length = int(required, c_int)
        if (mime_capacity > 0) then
            mime_length = min(len_trim(mime), int(mime_capacity) - 1)
            do i = 1, mime_length
                content_type(i) = mime(i:i)
            end do
            content_type(mime_length + 1) = c_null_char
        end if
        whisper_copy_result = 0_c_int
    end function whisper_copy_result

    function whisper_json_escape(text) result(escaped)
        character(len=*), intent(in) :: text
        type(string_t) :: escaped
        integer :: i, code
        character(len=6) :: hex

        call escaped%clear()
        do i = 1, len(text)
            code = iachar(text(i:i))
            select case (code)
            case (34); call escaped%append('\"')
            case (92); call escaped%append('\\')
            case (8); call escaped%append('\b')
            case (9); call escaped%append('\t')
            case (10); call escaped%append('\n')
            case (12); call escaped%append('\f')
            case (13); call escaped%append('\r')
            case (0:7, 11, 14:31)
                write(hex, '("\\u",z4.4)') code
                call escaped%append(hex)
            case default
                call escaped%append_char(text(i:i))
            end select
        end do
    end function whisper_json_escape

    function whisper_lower_ascii(text) result(lowered)
        character(len=*), intent(in) :: text
        character(len=:), allocatable :: lowered
        integer :: i, code

        allocate(character(len=len(text)) :: lowered)
        do i = 1, len(text)
            code = iachar(text(i:i))
            if (code >= iachar('A') .and. code <= iachar('Z')) then
                lowered(i:i) = achar(code + 32)
            else
                lowered(i:i) = text(i:i)
            end if
        end do
    end function whisper_lower_ascii

    integer function whisper_json_value_position(text, key)
        character(len=*), intent(in) :: text, key
        character(len=:), allocatable :: lowered, needle
        integer :: position, colon

        whisper_json_value_position = 0
        lowered = whisper_lower_ascii(text)
        needle = '"' // whisper_lower_ascii(key) // '"'
        position = index(lowered, needle)
        if (position <= 0) return
        if (position + len(key) + 2 > len(text)) return
        colon = index(text(position + len(key) + 2:), ':')
        if (colon <= 0) return
        whisper_json_value_position = position + len(key) + 2 + colon
    end function whisper_json_value_position

    function whisper_json_string(text, key, fallback, found) result(value)
        character(len=*), intent(in) :: text, key, fallback
        logical, intent(out) :: found
        character(len=:), allocatable :: value
        type(string_t) :: parsed
        integer :: first, position, code
        logical :: escaped, closed

        value = fallback
        found = .false.
        first = whisper_json_value_position(text, key)
        if (first <= 0) return
        do while (first <= len(text))
            code = iachar(text(first:first))
            if (code /= iachar(' ') .and. code /= iachar(char(9)) .and. code /= iachar(char(10)) .and. &
                code /= iachar(char(13))) exit
            first = first + 1
        end do
        if (first > len(text)) return
        if (text(first:first) /= '"') return
        call parsed%clear()
        escaped = .false.
        closed = .false.
        position = first + 1
        do while (position <= len(text))
            if (escaped) then
                select case (text(position:position))
                case ('"', '\', '/')
                    call parsed%append_char(text(position:position))
                case ('b'); call parsed%append_char(char(8))
                case ('f'); call parsed%append_char(char(12))
                case ('n'); call parsed%append_char(char(10))
                case ('r'); call parsed%append_char(char(13))
                case ('t'); call parsed%append_char(char(9))
                case default; call parsed%append_char(text(position:position))
                end select
                escaped = .false.
            else if (text(position:position) == '\') then
                escaped = .true.
            else if (text(position:position) == '"') then
                closed = .true.
                exit
            else
                call parsed%append_char(text(position:position))
            end if
            position = position + 1
        end do
        if (.not. closed .or. escaped) return
        value = parsed%as_character()
        found = .true.
    end function whisper_json_string

    subroutine whisper_form_value(raw_body, content_type, field, value, found)
        character(len=*), intent(in) :: raw_body, content_type, field
        character(len=:), allocatable, intent(out) :: value
        logical, intent(out) :: found
        character(len=:), allocatable :: lowered_type, boundary, marker, needle
        integer :: boundary_pos, start, stop, marker_pos, name_pos, candidate_pos, relative_pos
        integer :: header_end, payload_start, payload_end

        value = ''
        found = .false.
        lowered_type = whisper_lower_ascii(content_type)
        if (index(lowered_type, 'multipart/form-data') /= 1) return
        boundary_pos = index(lowered_type, 'boundary=')
        if (boundary_pos <= 0) return
        start = boundary_pos + len('boundary=')
        stop = len_trim(content_type)
        if (start > stop) return
        if (content_type(start:start) == '"') then
            start = start + 1
            if (start > stop) return
            if (content_type(stop:stop) == '"') stop = stop - 1
        end if
        if (stop < start) return
        boundary = content_type(start:stop)
        marker = '--' // boundary
        needle = 'name="' // field // '"'
        name_pos = index(raw_body, needle)
        if (name_pos <= 0) then
            needle = 'name=' // field
            name_pos = index(raw_body, needle)
        end if
        if (name_pos <= 0) return
        ! Locate the multipart delimiter preceding the field header.  Searching
        ! the prefix keeps a filename or binary payload from being mistaken for
        ! a form field with the same name.
        marker_pos = 0
        candidate_pos = 1
        do while (candidate_pos <= name_pos)
            relative_pos = index(raw_body(candidate_pos:name_pos), marker)
            if (relative_pos <= 0) exit
            marker_pos = candidate_pos + relative_pos - 1
            candidate_pos = marker_pos + len(marker)
        end do
        if (marker_pos <= 0) return
        header_end = index(raw_body(marker_pos:), char(13) // char(10) // char(13) // char(10))
        if (header_end <= 0) return
        payload_start = marker_pos + header_end + 3
        if (payload_start > len(raw_body)) return
        payload_end = index(raw_body(payload_start:), char(13) // char(10) // marker)
        if (payload_end <= 0) return
        payload_end = payload_start + payload_end - 2
        if (payload_end < payload_start) return
        value = trim(raw_body(payload_start:payload_end))
        found = .true.
    end subroutine whisper_form_value

    integer(int32) function whisper_text_int(value, fallback, found)
        character(len=*), intent(in) :: value
        integer(int32), intent(in) :: fallback
        logical, intent(out) :: found
        integer :: ios

        whisper_text_int = fallback
        read(value, *, iostat=ios) whisper_text_int
        found = ios == 0
        if (.not. found) whisper_text_int = fallback
    end function whisper_text_int

    integer(int64) function whisper_text_int64(value, fallback, found)
        character(len=*), intent(in) :: value
        integer(int64), intent(in) :: fallback
        logical, intent(out) :: found
        integer :: ios

        whisper_text_int64 = fallback
        read(value, *, iostat=ios) whisper_text_int64
        found = ios == 0
        if (.not. found) whisper_text_int64 = fallback
    end function whisper_text_int64

    real(real32) function whisper_text_real(value, fallback, found)
        character(len=*), intent(in) :: value
        real(real32), intent(in) :: fallback
        logical, intent(out) :: found
        integer :: ios

        whisper_text_real = fallback
        read(value, *, iostat=ios) whisper_text_real
        found = ios == 0
        if (.not. found) whisper_text_real = fallback
    end function whisper_text_real

    logical function finite_real32(value)
        real(real32), intent(in) :: value
        integer(int32) :: bits
        integer(int32), parameter :: exponent_mask = int(z'7f800000', int32)

        bits = transfer(value, bits)
        finite_real32 = iand(bits, exponent_mask) /= exponent_mask
    end function finite_real32

    integer(int32) function whisper_json_int(text, key, fallback, found)
        character(len=*), intent(in) :: text, key
        integer(int32), intent(in) :: fallback
        logical, intent(out) :: found
        integer :: position, colon, first, last, ios
        character(len=:), allocatable :: value

        whisper_json_int = fallback; found = .false.
        first = whisper_json_value_position(text, key)
        if (first <= 0) return
        do while (first <= len(text))
            if (text(first:first) /= ' ' .and. text(first:first) /= char(9)) exit
            first = first + 1
        end do
        last = first
        do while (last <= len(text))
            if (index('0123456789-', text(last:last)) <= 0) exit
            last = last + 1
        end do
        if (last <= first) return
        value = text(first:last - 1)
        read(value, *, iostat=ios) whisper_json_int
        found = ios == 0
        if (.not. found) whisper_json_int = fallback
    end function whisper_json_int

    integer(int64) function whisper_json_int64(text, key, fallback, found)
        character(len=*), intent(in) :: text, key
        integer(int64), intent(in) :: fallback
        logical, intent(out) :: found
        integer :: position, colon, first, last, ios
        character(len=:), allocatable :: value

        whisper_json_int64 = fallback; found = .false.
        first = whisper_json_value_position(text, key)
        if (first <= 0) return
        do while (first <= len(text))
            if (text(first:first) /= ' ' .and. text(first:first) /= char(9)) exit
            first = first + 1
        end do
        last = first
        do while (last <= len(text))
            if (index('0123456789-', text(last:last)) <= 0) exit
            last = last + 1
        end do
        if (last <= first) return
        value = text(first:last - 1)
        read(value, *, iostat=ios) whisper_json_int64
        found = ios == 0
        if (.not. found) whisper_json_int64 = fallback
    end function whisper_json_int64

    real(real32) function whisper_json_real(text, key, fallback, found)
        character(len=*), intent(in) :: text, key
        real(real32), intent(in) :: fallback
        logical, intent(out) :: found
        integer :: position, colon, first, last, ios
        character(len=:), allocatable :: value

        whisper_json_real = fallback; found = .false.
        first = whisper_json_value_position(text, key)
        if (first <= 0) return
        do while (first <= len(text))
            if (text(first:first) /= ' ' .and. text(first:first) /= char(9)) exit
            first = first + 1
        end do
        last = first
        do while (last <= len(text))
            if (index('0123456789+-.eE', text(last:last)) <= 0) exit
            last = last + 1
        end do
        if (last <= first) return
        value = text(first:last - 1)
        read(value, *, iostat=ios) whisper_json_real
        found = ios == 0
        if (.not. found) whisper_json_real = fallback
    end function whisper_json_real

end module fortai_whisper_service
