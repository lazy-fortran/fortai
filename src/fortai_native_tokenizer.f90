module fortai_native_tokenizer
    use, intrinsic :: iso_fortran_env, only: int32, int64
    use fortai_gguf_runtime, only: gguf_file_t
    implicit none
    private

    type :: piece_t
        character(len=:), allocatable :: text
    end type piece_t

    type, public :: fortai_native_tokenizer_t
        integer(int32) :: vocab_size = 0_int32
        integer(int32) :: bos_token = -1_int32
        integer(int32) :: eos_token = -1_int32
        logical :: add_bos_token = .false.
        ! Qwen ships both thinking-default and no-thinking-default chat
        ! templates.  Keep the template policy with the tokenizer rather
        ! than guessing from the model filename in the HTTP layer.
        logical :: default_enable_thinking = .true.
        logical :: supports_reasoning_effort = .false.
        logical :: supports_preserve_thinking = .false.
        type(piece_t), allocatable :: vocabulary(:)
        type(piece_t), allocatable :: merge_left(:)
        type(piece_t), allocatable :: merge_right(:)
    contains
        procedure :: close => native_tokenizer_close
        procedure :: open => native_tokenizer_open
        procedure :: encode => native_tokenizer_encode
        procedure :: decode => native_tokenizer_decode
        procedure :: is_eos => native_tokenizer_is_eos
        procedure :: is_stop => native_tokenizer_is_stop
    end type fortai_native_tokenizer_t

contains

    subroutine native_tokenizer_open(self, file, stat_ok)
        class(fortai_native_tokenizer_t), intent(inout) :: self
        type(gguf_file_t), intent(in) :: file
        logical, intent(out) :: stat_ok
        integer :: metadata_index, i, n, max_length, separator
        character(len=:), allocatable :: merge, chat_template

        call self%close()
        stat_ok = .false.
        self%bos_token = int(file%meta_int('tokenizer.ggml.bos_token_id', -1_int64), int32)
        self%eos_token = int(file%meta_int('tokenizer.ggml.eos_token_id', -1_int64), int32)
        ! GGUF does not require BOS on every encoded string.  In particular,
        ! Qwen3.5's tokenizer omits tokenizer.ggml.add_bos_token and llama.cpp
        ! therefore starts a chat prompt directly with <|im_start|>.  Treat a
        ! missing policy as false; only an explicit GGUF boolean enables BOS.
        metadata_index = file%find_meta('tokenizer.ggml.add_bos_token')
        if (metadata_index > 0) then
            self%add_bos_token = file%metadata(metadata_index)%value_type == 7 .and. &
                file%metadata(metadata_index)%logical_value
        end if
        metadata_index = file%find_meta('tokenizer.chat_template')
        if (metadata_index > 0) then
            if (file%metadata(metadata_index)%value_type == 8) then
                if (allocated(file%metadata(metadata_index)%string_value)) then
                    chat_template = file%metadata(metadata_index)%string_value
                    ! Qwen3.5 no-thinking templates test for `is true`;
                    ! thinking templates test for `is false` (or explicitly
                    ! allow undefined).
                    if (index(chat_template, 'enable_thinking is defined and enable_thinking is true') > 0 .and. &
                        index(chat_template, 'enable_thinking is undefined or enable_thinking is true') == 0) then
                        self%default_enable_thinking = .false.
                    else
                        self%default_enable_thinking = .true.
                    end if
                    self%supports_reasoning_effort = index(chat_template, 'reasoning_effort') > 0
                    self%supports_preserve_thinking = index(chat_template, 'preserve_thinking') > 0
                end if
            end if
        end if
        metadata_index = file%find_meta('tokenizer.ggml.tokens')
        if (metadata_index == 0) return
        if (.not. allocated(file%metadata(metadata_index)%string_values)) return
        n = size(file%metadata(metadata_index)%string_values)
        if (n <= 0) return
        max_length = 1
        do i = 1, n
            max_length = max(max_length, len(file%metadata(metadata_index)%string_values(i)%value))
        end do
        allocate(self%vocabulary(n))
        do i = 1, n
            self%vocabulary(i)%text = file%metadata(metadata_index)%string_values(i)%value
        end do
        self%vocab_size = int(n, int32)

        metadata_index = file%find_meta('tokenizer.ggml.merges')
        if (metadata_index > 0) then
            if (allocated(file%metadata(metadata_index)%string_values)) then
                n = size(file%metadata(metadata_index)%string_values)
                allocate(self%merge_left(n), self%merge_right(n))
                do i = 1, n
                    merge = file%metadata(metadata_index)%string_values(i)%value
                    separator = index(merge, ' ')
                    if (separator <= 1 .or. separator >= len(merge)) cycle
                    self%merge_left(i)%text = merge(:separator - 1)
                    self%merge_right(i)%text = merge(separator + 1:)
                end do
            end if
        end if
        stat_ok = allocated(self%vocabulary)
    end subroutine native_tokenizer_open

    subroutine native_tokenizer_close(self)
        class(fortai_native_tokenizer_t), intent(inout) :: self
        integer :: i
        if (allocated(self%vocabulary)) then
            do i = 1, size(self%vocabulary)
                if (allocated(self%vocabulary(i)%text)) deallocate(self%vocabulary(i)%text)
            end do
            deallocate(self%vocabulary)
        end if
        if (allocated(self%merge_left)) then
            do i = 1, size(self%merge_left)
                if (allocated(self%merge_left(i)%text)) deallocate(self%merge_left(i)%text)
                if (allocated(self%merge_right(i)%text)) deallocate(self%merge_right(i)%text)
            end do
            deallocate(self%merge_left, self%merge_right)
        end if
        self%vocab_size = 0_int32
        self%bos_token = -1_int32
        self%eos_token = -1_int32
        self%add_bos_token = .false.
        self%default_enable_thinking = .true.
        self%supports_reasoning_effort = .false.
        self%supports_preserve_thinking = .false.
    end subroutine native_tokenizer_close

    integer function lookup_vocabulary(self, text)
        class(fortai_native_tokenizer_t), intent(in) :: self
        character(len=*), intent(in) :: text
        integer :: i
        lookup_vocabulary = -1
        if (.not. allocated(self%vocabulary)) return
        do i = 1, size(self%vocabulary)
            if (allocated(self%vocabulary(i)%text)) then
                if (self%vocabulary(i)%text == text) then
                    lookup_vocabulary = i - 1
                    return
                end if
            end if
        end do
    end function lookup_vocabulary

    integer function lookup_merge(self, left, right)
        class(fortai_native_tokenizer_t), intent(in) :: self
        character(len=*), intent(in) :: left, right
        integer :: i
        lookup_merge = huge(0)
        if (.not. allocated(self%merge_left)) return
        do i = 1, size(self%merge_left)
            if (.not. allocated(self%merge_left(i)%text)) cycle
            if (self%merge_left(i)%text == left .and. self%merge_right(i)%text == right) then
                lookup_merge = i
                return
            end if
        end do
    end function lookup_merge

    pure logical function ascii_word_byte(value)
        integer, intent(in) :: value
        ascii_word_byte = (value >= iachar('A') .and. value <= iachar('Z')) .or. &
            (value >= iachar('a') .and. value <= iachar('z')) .or. &
            (value >= iachar('0') .and. value <= iachar('9')) .or. value >= 128
    end function ascii_word_byte

    integer function byte_unicode(value)
        integer, intent(in) :: value
        integer :: b, extra
        logical :: listed
        extra = 0
        do b = 0, 255
            listed = (b >= 33 .and. b <= 126) .or. (b >= 161 .and. b <= 172) .or. &
                (b >= 174 .and. b <= 255)
            if (.not. listed) then
                if (b == value) then
                    byte_unicode = 256 + extra
                    return
                end if
                extra = extra + 1
            else if (b == value) then
                byte_unicode = b
                return
            end if
        end do
        byte_unicode = value
    end function byte_unicode

    function utf8_encode(codepoint) result(text)
        integer, intent(in) :: codepoint
        character(len=:), allocatable :: text
        if (codepoint <= int(z'7f')) then
            allocate(character(len=1) :: text)
            text(1:1) = achar(codepoint)
        else if (codepoint <= int(z'7ff')) then
            allocate(character(len=2) :: text)
            text(1:1) = achar(int(z'c0') + ishft(codepoint, -6))
            text(2:2) = achar(int(z'80') + iand(codepoint, int(z'3f')))
        else if (codepoint <= int(z'ffff')) then
            allocate(character(len=3) :: text)
            text(1:1) = achar(int(z'e0') + ishft(codepoint, -12))
            text(2:2) = achar(int(z'80') + iand(ishft(codepoint, -6), int(z'3f')))
            text(3:3) = achar(int(z'80') + iand(codepoint, int(z'3f')))
        else
            allocate(character(len=4) :: text)
            text(1:1) = achar(int(z'f0') + ishft(codepoint, -18))
            text(2:2) = achar(int(z'80') + iand(ishft(codepoint, -12), int(z'3f')))
            text(3:3) = achar(int(z'80') + iand(ishft(codepoint, -6), int(z'3f')))
            text(4:4) = achar(int(z'80') + iand(codepoint, int(z'3f')))
        end if
    end function utf8_encode

    subroutine append_symbol(symbols, count, value)
        type(piece_t), allocatable, intent(inout) :: symbols(:)
        integer, intent(inout) :: count
        character(len=*), intent(in) :: value
        type(piece_t), allocatable :: grown(:)
        integer :: capacity
        if (.not. allocated(symbols)) then
            allocate(symbols(16))
        else if (count == size(symbols)) then
            capacity = max(16, 2 * size(symbols))
            allocate(grown(capacity))
            if (count > 0) grown(:count) = symbols(:count)
            call move_alloc(grown, symbols)
        end if
        count = count + 1
        symbols(count)%text = value
    end subroutine append_symbol

    subroutine encode_piece(self, raw, output, count, capacity)
        class(fortai_native_tokenizer_t), intent(in) :: self
        character(len=*), intent(in) :: raw
        integer(int32), intent(inout) :: output(:)
        integer, intent(inout) :: count, capacity
        type(piece_t), allocatable :: symbols(:)
        integer :: i, n, best, rank, candidate, id, value
        character(len=:), allocatable :: merged

        n = 0
        i = 1
        do while (i <= len(raw))
            value = byte_unicode(iachar(raw(i:i)))
            call append_symbol(symbols, n, utf8_encode(value))
            i = i + 1
        end do
        do while (n > 1)
            best = 0
            rank = huge(0)
            do i = 1, n - 1
                candidate = lookup_merge(self, symbols(i)%text, symbols(i + 1)%text)
                if (candidate < rank) then
                    rank = candidate
                    best = i
                end if
            end do
            if (best == 0) exit
            merged = symbols(best)%text // symbols(best + 1)%text
            symbols(best)%text = merged
            do i = best + 1, n - 1
                symbols(i) = symbols(i + 1)
            end do
            n = n - 1
        end do
        do i = 1, n
            id = lookup_vocabulary(self, symbols(i)%text)
            if (id < 0) id = lookup_vocabulary(self, '<unk>')
            if (id >= 0) then
                if (count < capacity) then
                    count = count + 1
                    output(count) = int(id, int32)
                end if
            end if
        end do
    end subroutine encode_piece

    subroutine encode_plain(self, text, output, count, capacity)
        class(fortai_native_tokenizer_t), intent(in) :: self
        character(len=*), intent(in) :: text
        integer(int32), intent(inout) :: output(:)
        integer, intent(inout) :: count, capacity
        integer :: start, finish, n, value
        character(len=:), allocatable :: piece
        start = 1
        do while (start <= len(text))
            finish = start
            if (iachar(text(start:start)) == iachar(' ')) then
                if (start < len(text)) then
                    if (ascii_word_byte(iachar(text(start + 1:start + 1)))) finish = finish + 1
                end if
            end if
            if (ascii_word_byte(iachar(text(finish:finish)))) then
                do while (finish < len(text))
                    if (.not. ascii_word_byte(iachar(text(finish + 1:finish + 1)))) exit
                    finish = finish + 1
                end do
            else if (iachar(text(start:start)) == iachar(' ') .or. &
                    iachar(text(start:start)) == iachar(new_line('a'))) then
                do while (finish < len(text))
                    if (iachar(text(finish + 1:finish + 1)) /= iachar(text(start:start))) exit
                    finish = finish + 1
                end do
            else
                do while (finish < len(text))
                    if (ascii_word_byte(iachar(text(finish + 1:finish + 1)))) exit
                    if (iachar(text(finish + 1:finish + 1)) == iachar(' ')) exit
                    if (iachar(text(finish + 1:finish + 1)) == iachar(new_line('a'))) exit
                    finish = finish + 1
                end do
            end if
            piece = text(start:finish)
            call encode_piece(self, piece, output, count, capacity)
            start = finish + 1
        end do
    end subroutine encode_plain

    subroutine native_tokenizer_encode(self, text, ids)
        class(fortai_native_tokenizer_t), intent(in) :: self
        character(len=*), intent(in) :: text
        integer(int32), allocatable, intent(out) :: ids(:)
        integer(int32), allocatable :: work(:)
        integer :: capacity, count, start, finish, relative, special_end, id
        integer :: marker_length, candidate, marker_position
        character(len=:), allocatable :: special
        character(len=16), parameter :: inline_specials(6) = [character(len=16) :: &
            '<think>', '</think>', '<tool_call>', '</tool_call>', '<tool_response>', '</tool_response>']

        capacity = max(32, 4 * len(text) + 8)
        allocate(work(capacity))
        count = 0
        if (self%add_bos_token .and. self%bos_token >= 0) then
            count = 1
            work(count) = self%bos_token
        end if
        start = 1
        do while (start <= len(text))
            ! Added-token vocabularies carry Qwen's inline reasoning/tool tags
            ! as atomic tokens even though they are not delimited by <|...|>.
            ! Treat them like the existing control tokens; BPE over their
            ! characters would otherwise add two tokens per tag and change the
            ! chat prompt (notably making no-thinking requests stop at EOS).
            relative = index(text(start:), '<|')
            marker_length = 0
            do candidate = 1, size(inline_specials)
                marker_position = index(text(start:), trim(inline_specials(candidate)))
                if (marker_position > 0 .and. (relative == 0 .or. marker_position < relative)) then
                    relative = marker_position
                    marker_length = len_trim(inline_specials(candidate))
                end if
            end do
            if (relative == 0) then
                call encode_plain(self, text(start:), work, count, capacity)
                exit
            end if
            finish = start + relative - 1
            if (finish > start) call encode_plain(self, text(start:finish - 1), work, count, capacity)
            if (marker_length > 0) then
                special_end = finish + marker_length - 1
            else
                special_end = 0
                if (finish + 2 <= len(text)) then
                    special_end = index(text(finish + 2:), '|>')
                    if (special_end > 0) special_end = finish + 2 + special_end
                end if
                if (special_end == 0) then
                    call encode_plain(self, text(finish:), work, count, capacity)
                    exit
                end if
            end if
            special = text(finish:special_end)
            id = lookup_vocabulary(self, special)
            if (id < 0) then
                call encode_plain(self, special, work, count, capacity)
            else
                if (count < capacity) then
                    count = count + 1
                    work(count) = int(id, int32)
                end if
            end if
            start = special_end + 1
        end do
        if (allocated(ids)) deallocate(ids)
        allocate(ids(count))
        if (count > 0) ids = work(:count)
        deallocate(work)
    end subroutine native_tokenizer_encode

    subroutine native_tokenizer_decode(self, ids, text)
        class(fortai_native_tokenizer_t), intent(in) :: self
        integer(int32), intent(in) :: ids(:)
        character(len=:), allocatable, intent(out) :: text
        integer :: i, j, codepoint, width, byte_value, total
        character(len=:), allocatable :: piece
        total = 0
        do i = 1, size(ids)
            if (ids(i) < 0 .or. ids(i) >= self%vocab_size) cycle
            piece = self%vocabulary(ids(i) + 1)%text
            j = 1
            do while (j <= len(piece))
                call utf8_decode(piece, j, codepoint, width)
                if (codepoint >= 256 .and. codepoint < 512) then
                    byte_value = codepoint - 256
                else if (codepoint <= 255) then
                    byte_value = codepoint
                else
                    byte_value = 32
                end if
                total = total + 1
                j = j + width
            end do
        end do
        allocate(character(len=total) :: text)
        total = 0
        do i = 1, size(ids)
            if (ids(i) < 0 .or. ids(i) >= self%vocab_size) cycle
            piece = self%vocabulary(ids(i) + 1)%text
            j = 1
            do while (j <= len(piece))
                call utf8_decode(piece, j, codepoint, width)
                if (codepoint >= 256 .and. codepoint < 512) then
                    byte_value = codepoint - 256
                else if (codepoint <= 255) then
                    byte_value = codepoint
                else
                    byte_value = 32
                end if
                total = total + 1
                text(total:total) = achar(byte_value)
                j = j + width
            end do
        end do
    end subroutine native_tokenizer_decode

    subroutine utf8_decode(text, start, codepoint, width)
        character(len=*), intent(in) :: text
        integer, intent(in) :: start
        integer, intent(out) :: codepoint, width
        integer :: first
        first = iachar(text(start:start))
        if (first < 128) then
            codepoint = first; width = 1
        else if (first < 224 .and. start + 1 <= len(text)) then
            codepoint = iand(first, 31) * 64 + iand(iachar(text(start + 1:start + 1)), 63); width = 2
        else if (first < 240 .and. start + 2 <= len(text)) then
            codepoint = iand(first, 15) * 4096 + iand(iachar(text(start + 1:start + 1)), 63) * 64 + &
                iand(iachar(text(start + 2:start + 2)), 63); width = 3
        else if (start + 3 <= len(text)) then
            codepoint = iand(first, 7) * 262144 + iand(iachar(text(start + 1:start + 1)), 63) * 4096 + &
                iand(iachar(text(start + 2:start + 2)), 63) * 64 + iand(iachar(text(start + 3:start + 3)), 63); width = 4
        else
            codepoint = first; width = 1
        end if
    end subroutine utf8_decode

    logical function native_tokenizer_is_eos(self, token)
        class(fortai_native_tokenizer_t), intent(in) :: self
        integer(int32), intent(in) :: token
        native_tokenizer_is_eos = token == self%eos_token
    end function native_tokenizer_is_eos

    logical function native_tokenizer_is_stop(self, token)
        class(fortai_native_tokenizer_t), intent(in) :: self
        integer(int32), intent(in) :: token
        integer :: index

        native_tokenizer_is_stop = token == self%eos_token
        if (native_tokenizer_is_stop) return
        index = int(token) + 1
        if (index < 1 .or. .not. allocated(self%vocabulary)) return
        if (index > size(self%vocabulary)) return
        if (.not. allocated(self%vocabulary(index)%text)) return
        native_tokenizer_is_stop = self%vocabulary(index)%text == '<|im_end|>' .or. &
            self%vocabulary(index)%text == '<|endoftext|>'
    end function native_tokenizer_is_stop

end module fortai_native_tokenizer
