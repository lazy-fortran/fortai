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
        logical :: qwen35_pretokenizer = .false.
        ! llama.cpp defines enable_thinking=true whenever the loaded chat
        ! template supports that control.  Keep the same native default;
        ! an explicit chat_template_kwargs.enable_thinking=false still
        ! selects the template's no-thinking branch.
        logical :: default_enable_thinking = .true.
        logical :: supports_reasoning_effort = .false.
        logical :: supports_preserve_thinking = .false.
        ! GGUF vocabularies are large (Qwen3.8 has ~248k entries).  Keeping
        ! every piece as a separately allocated deferred-length character
        ! object makes startup allocator-bound and needlessly fragments RSS.
        ! Store the immutable text in compact contiguous blobs instead.
        character(len=:), allocatable :: vocabulary_blob
        integer(int32), allocatable :: vocabulary_offset(:)
        integer(int32), allocatable :: vocabulary_length(:)
        character(len=:), allocatable :: merge_left_blob
        character(len=:), allocatable :: merge_right_blob
        integer(int32), allocatable :: merge_left_offset(:)
        integer(int32), allocatable :: merge_left_length(:)
        integer(int32), allocatable :: merge_right_offset(:)
        integer(int32), allocatable :: merge_right_length(:)
    contains
        procedure :: close => native_tokenizer_close
        procedure :: open => native_tokenizer_open
        procedure :: encode => native_tokenizer_encode
        procedure :: decode => native_tokenizer_decode
        procedure :: token_piece => native_tokenizer_token_piece
        procedure :: is_eos => native_tokenizer_is_eos
        procedure :: is_stop => native_tokenizer_is_stop
    end type fortai_native_tokenizer_t

contains

    subroutine native_tokenizer_open(self, file, stat_ok)
        class(fortai_native_tokenizer_t), intent(inout) :: self
        type(gguf_file_t), intent(in) :: file
        logical, intent(out) :: stat_ok
        integer :: metadata_index, i, n, separator, length
        integer(int64) :: total_length, total_left_length, total_right_length, position
        character(len=:), allocatable :: chat_template

        call self%close()
        stat_ok = .false.
        self%bos_token = int(file%meta_int('tokenizer.ggml.bos_token_id', -1_int64), int32)
        self%eos_token = int(file%meta_int('tokenizer.ggml.eos_token_id', -1_int64), int32)
        metadata_index = file%find_meta('tokenizer.ggml.pre')
        if (metadata_index > 0) then
            if (file%metadata(metadata_index)%value_type == 8) then
                if (allocated(file%metadata(metadata_index)%string_value)) then
                    self%qwen35_pretokenizer = trim(file%metadata(metadata_index)%string_value) == 'qwen35'
                end if
            end if
        end if
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
                    ! Do not follow the template's undefined-value branch:
                    ! llama.cpp supplies a true value before rendering.
                    self%default_enable_thinking = .true.
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
        total_length = 0_int64
        do i = 1, n
            total_length = total_length + len(file%metadata(metadata_index)%string_values(i)%value)
        end do
        allocate(character(len=max(1_int64, total_length)) :: self%vocabulary_blob)
        allocate(self%vocabulary_offset(n), self%vocabulary_length(n))
        position = 1_int64
        do i = 1, n
            length = len(file%metadata(metadata_index)%string_values(i)%value)
            self%vocabulary_offset(i) = int(position, int32)
            self%vocabulary_length(i) = int(length, int32)
            if (length > 0) self%vocabulary_blob(position:position + length - 1) = &
                file%metadata(metadata_index)%string_values(i)%value
            position = position + length
        end do
        self%vocab_size = int(n, int32)

        metadata_index = file%find_meta('tokenizer.ggml.merges')
        if (metadata_index > 0) then
            if (allocated(file%metadata(metadata_index)%string_values)) then
                n = size(file%metadata(metadata_index)%string_values)
                total_left_length = 0_int64
                total_right_length = 0_int64
                do i = 1, n
                    separator = index(file%metadata(metadata_index)%string_values(i)%value, ' ')
                    if (separator <= 1 .or. separator >= len(file%metadata(metadata_index)%string_values(i)%value)) cycle
                    total_left_length = total_left_length + separator - 1
                    total_right_length = total_right_length + &
                        len(file%metadata(metadata_index)%string_values(i)%value) - separator
                end do
                allocate(character(len=max(1_int64, total_left_length)) :: self%merge_left_blob)
                allocate(character(len=max(1_int64, total_right_length)) :: self%merge_right_blob)
                allocate(self%merge_left_offset(n), self%merge_left_length(n), &
                    self%merge_right_offset(n), self%merge_right_length(n))
                self%merge_left_offset = 0_int32
                self%merge_left_length = 0_int32
                self%merge_right_offset = 0_int32
                self%merge_right_length = 0_int32
                position = 1_int64
                total_right_length = 1_int64
                do i = 1, n
                    separator = index(file%metadata(metadata_index)%string_values(i)%value, ' ')
                    if (separator <= 1 .or. separator >= len(file%metadata(metadata_index)%string_values(i)%value)) cycle
                    length = separator - 1
                    self%merge_left_offset(i) = int(position, int32)
                    self%merge_left_length(i) = int(length, int32)
                    self%merge_left_blob(position:position + length - 1) = &
                        file%metadata(metadata_index)%string_values(i)%value(:length)
                    position = position + length
                    length = len(file%metadata(metadata_index)%string_values(i)%value) - separator
                    self%merge_right_offset(i) = int(total_right_length, int32)
                    self%merge_right_length(i) = int(length, int32)
                    self%merge_right_blob(total_right_length:total_right_length + length - 1) = &
                        file%metadata(metadata_index)%string_values(i)%value(separator + 1:)
                    total_right_length = total_right_length + length
                end do
            end if
        end if
        stat_ok = allocated(self%vocabulary_offset)
    end subroutine native_tokenizer_open

    subroutine native_tokenizer_close(self)
        class(fortai_native_tokenizer_t), intent(inout) :: self
        if (allocated(self%vocabulary_blob)) deallocate(self%vocabulary_blob)
        if (allocated(self%vocabulary_offset)) deallocate(self%vocabulary_offset)
        if (allocated(self%vocabulary_length)) deallocate(self%vocabulary_length)
        if (allocated(self%merge_left_blob)) deallocate(self%merge_left_blob)
        if (allocated(self%merge_right_blob)) deallocate(self%merge_right_blob)
        if (allocated(self%merge_left_offset)) deallocate(self%merge_left_offset)
        if (allocated(self%merge_left_length)) deallocate(self%merge_left_length)
        if (allocated(self%merge_right_offset)) deallocate(self%merge_right_offset)
        if (allocated(self%merge_right_length)) deallocate(self%merge_right_length)
        self%vocab_size = 0_int32
        self%bos_token = -1_int32
        self%eos_token = -1_int32
        self%add_bos_token = .false.
        self%qwen35_pretokenizer = .false.
        self%default_enable_thinking = .true.
        self%supports_reasoning_effort = .false.
        self%supports_preserve_thinking = .false.
    end subroutine native_tokenizer_close

    integer function lookup_vocabulary(self, text)
        class(fortai_native_tokenizer_t), intent(in) :: self
        character(len=*), intent(in) :: text
        integer :: i
        lookup_vocabulary = -1
        if (.not. allocated(self%vocabulary_offset)) return
        do i = 1, size(self%vocabulary_offset)
            if (self%vocabulary_length(i) /= len(text)) cycle
            if (len(text) == 0) then
                lookup_vocabulary = i - 1
                return
            else if (self%vocabulary_blob(self%vocabulary_offset(i): &
                    self%vocabulary_offset(i) + self%vocabulary_length(i) - 1) == text) then
                lookup_vocabulary = i - 1
                return
            end if
        end do
    end function lookup_vocabulary

    integer function lookup_merge(self, left, right)
        class(fortai_native_tokenizer_t), intent(in) :: self
        character(len=*), intent(in) :: left, right
        integer :: i
        lookup_merge = huge(0)
        if (.not. allocated(self%merge_left_offset)) return
        do i = 1, size(self%merge_left_offset)
            if (self%merge_left_length(i) /= len(left) .or. self%merge_right_length(i) /= len(right)) cycle
            if (len(left) > 0) then
                if (self%merge_left_blob(self%merge_left_offset(i): &
                    self%merge_left_offset(i) + self%merge_left_length(i) - 1) /= left) cycle
            end if
            if (len(right) > 0) then
                if (self%merge_right_blob(self%merge_right_offset(i): &
                    self%merge_right_offset(i) + self%merge_right_length(i) - 1) /= right) cycle
            end if
            if (self%merge_left_length(i) == len(left) .and. self%merge_right_length(i) == len(right)) then
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
        if ((value >= 33 .and. value <= 126) .or. (value >= 161 .and. value <= 172) .or. &
            (value >= 174 .and. value <= 255)) then
            byte_unicode = value
            return
        end if
        if (value >= 0 .and. value <= 32) then
            byte_unicode = 256 + value
            return
        end if
        if (value >= 127 .and. value <= 160) then
            byte_unicode = value + 162
            return
        end if
        if (value == 173) then
            byte_unicode = 323
            return
        end if
        byte_unicode = value
    end function byte_unicode

    integer function unicode_byte(value)
        integer, intent(in) :: value

        unicode_byte = value
        if (value < 256 .or. value >= 512) return
        if (value <= 288) then
            unicode_byte = value - 256
        else if (value <= 322) then
            unicode_byte = value - 162
        else if (value == 323) then
            unicode_byte = 173
        end if
    end function unicode_byte

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

        if (self%qwen35_pretokenizer) then
            call encode_plain_qwen35(self, text, output, count, capacity)
            return
        end if
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

    subroutine encode_plain_qwen35(self, text, output, count, capacity)
        class(fortai_native_tokenizer_t), intent(in) :: self
        character(len=*), intent(in) :: text
        integer(int32), intent(inout) :: output(:)
        integer, intent(inout) :: count, capacity
        integer :: position, cursor, finish, width, codepoint
        integer :: next_width, next_codepoint, contraction_length
        integer :: scan_width, scan_codepoint
        logical :: prefix_word, prefix_punctuation

        ! This is the Qwen3.5 GPT-2 pre-tokenizer expressed as a small UTF-8
        ! scanner.  In particular, its optional leading character before a
        ! word includes tabs and punctuation ("\twith" is one pre-token), and
        ! numbers are split one Unicode code point at a time.  BPE itself is
        ! still performed by encode_piece, so the scanner only determines the
        ! merge boundaries.
        position = 1
        do while (position <= len(text))
            call utf8_decode(text, position, codepoint, width)
            finish = position + width - 1
            prefix_word = .false.
            prefix_punctuation = .false.
            if (qwen35_is_optional_prefix(codepoint)) then
                cursor = position + width
                if (cursor <= len(text)) then
                    call utf8_decode(text, cursor, next_codepoint, next_width)
                    if (qwen35_is_word(next_codepoint)) then
                        prefix_word = .true.
                    else if (codepoint == iachar(' ') .and. .not. qwen35_is_whitespace(next_codepoint) .and. &
                            .not. qwen35_is_number(next_codepoint)) then
                        prefix_punctuation = .true.
                    end if
                end if
            end if
            contraction_length = qwen35_contraction_length(text, position)
            if (contraction_length > 0) then
                finish = position + contraction_length - 1
            else if (qwen35_is_number(codepoint)) then
                continue
            else if (qwen35_is_word(codepoint)) then
                cursor = position + width
                do while (cursor <= len(text))
                    call utf8_decode(text, cursor, scan_codepoint, scan_width)
                    if (.not. qwen35_is_word(scan_codepoint)) exit
                    finish = cursor + scan_width - 1
                    cursor = finish + 1
                end do
            else if (prefix_word) then
                cursor = position + width
                call utf8_decode(text, cursor, next_codepoint, next_width)
                finish = cursor + next_width - 1
                cursor = finish + 1
                do while (cursor <= len(text))
                    call utf8_decode(text, cursor, scan_codepoint, scan_width)
                    if (.not. qwen35_is_word(scan_codepoint)) exit
                    finish = cursor + scan_width - 1
                    cursor = finish + 1
                end do
            else if (prefix_punctuation) then
                cursor = position + width
                finish = cursor - 1
                do while (cursor <= len(text))
                    call utf8_decode(text, cursor, scan_codepoint, scan_width)
                    if (qwen35_is_whitespace(scan_codepoint) .or. qwen35_is_word(scan_codepoint) .or. &
                        qwen35_is_number(scan_codepoint)) exit
                    finish = cursor + scan_width - 1
                    cursor = finish + 1
                end do
                if (cursor <= len(text)) then
                    call utf8_decode(text, cursor, scan_codepoint, scan_width)
                    if (qwen35_is_newline(scan_codepoint)) finish = cursor + scan_width - 1
                end if
            else if (qwen35_is_whitespace(codepoint)) then
                if (qwen35_is_newline(codepoint)) then
                    cursor = position + width
                    do while (cursor <= len(text))
                        call utf8_decode(text, cursor, scan_codepoint, scan_width)
                        if (.not. qwen35_is_newline(scan_codepoint)) exit
                        finish = cursor + scan_width - 1
                        cursor = finish + 1
                    end do
                else
                    cursor = position
                    do while (cursor <= len(text))
                        call utf8_decode(text, cursor, scan_codepoint, scan_width)
                        if (.not. qwen35_is_whitespace(scan_codepoint)) exit
                        if (qwen35_is_newline(scan_codepoint)) then
                            finish = cursor + scan_width - 1
                            cursor = finish + 1
                            do while (cursor <= len(text))
                                call utf8_decode(text, cursor, scan_codepoint, scan_width)
                                if (.not. qwen35_is_newline(scan_codepoint)) exit
                                finish = cursor + scan_width - 1
                                cursor = finish + 1
                            end do
                            exit
                        end if
                        finish = cursor + scan_width - 1
                        cursor = finish + 1
                    end do
                end if
            else
                ! Optional ASCII space before a punctuation run.
                cursor = position
                if (codepoint == iachar(' ')) then
                    cursor = position + width
                    if (cursor <= len(text)) then
                        call utf8_decode(text, cursor, next_codepoint, next_width)
                        if (qwen35_is_whitespace(next_codepoint) .or. qwen35_is_word(next_codepoint) .or. &
                            qwen35_is_number(next_codepoint)) then
                            cursor = position
                        end if
                    end if
                end if
                if (cursor > position) then
                    finish = cursor - 1
                end if
                do while (cursor <= len(text))
                    call utf8_decode(text, cursor, scan_codepoint, scan_width)
                    if (qwen35_is_whitespace(scan_codepoint) .or. qwen35_is_word(scan_codepoint) .or. &
                        qwen35_is_number(scan_codepoint)) exit
                    finish = cursor + scan_width - 1
                    cursor = finish + 1
                end do
                if (cursor <= len(text)) then
                    call utf8_decode(text, cursor, scan_codepoint, scan_width)
                    if (qwen35_is_newline(scan_codepoint) .or. scan_codepoint == iachar('/')) then
                        finish = cursor + scan_width - 1
                    end if
                end if
            end if
            call encode_piece(self, text(position:finish), output, count, capacity)
            position = finish + 1
        end do
    end subroutine encode_plain_qwen35

    integer function qwen35_contraction_length(text, position)
        character(len=*), intent(in) :: text
        integer, intent(in) :: position
        integer :: first, second

        qwen35_contraction_length = 0
        if (position > len(text)) return
        if (iachar(text(position:position)) /= iachar("'")) return
        if (position + 1 > len(text)) return
        first = qwen35_lower_ascii(iachar(text(position + 1:position + 1)))
        if (first == iachar('s') .or. first == iachar('t') .or. first == iachar('m') .or. &
            first == iachar('d')) then
            qwen35_contraction_length = 2
            return
        end if
        if (first == iachar('r') .or. first == iachar('v') .or. first == iachar('l')) then
            if (position + 2 > len(text)) return
            second = qwen35_lower_ascii(iachar(text(position + 2:position + 2)))
            if ((first == iachar('r') .or. first == iachar('v')) .and. second == iachar('e')) then
                qwen35_contraction_length = 3
            else if (first == iachar('l') .and. second == iachar('l')) then
                qwen35_contraction_length = 3
            end if
        end if
    end function qwen35_contraction_length

    integer function qwen35_lower_ascii(value)
        integer, intent(in) :: value
        qwen35_lower_ascii = value
        if (value >= iachar('A') .and. value <= iachar('Z')) qwen35_lower_ascii = value + 32
    end function qwen35_lower_ascii

    logical function qwen35_is_number(codepoint)
        integer, intent(in) :: codepoint

        qwen35_is_number = (codepoint >= int(z'30') .and. codepoint <= int(z'39')) .or. &
            (codepoint >= int(z'0660') .and. codepoint <= int(z'0669')) .or. &
            (codepoint >= int(z'06f0') .and. codepoint <= int(z'06f9')) .or. &
            (codepoint >= int(z'0966') .and. codepoint <= int(z'096f')) .or. &
            (codepoint >= int(z'09e6') .and. codepoint <= int(z'09ef')) .or. &
            (codepoint >= int(z'0a66') .and. codepoint <= int(z'0a6f')) .or. &
            (codepoint >= int(z'0ae6') .and. codepoint <= int(z'0aef')) .or. &
            (codepoint >= int(z'0b66') .and. codepoint <= int(z'0b6f')) .or. &
            (codepoint >= int(z'0be6') .and. codepoint <= int(z'0bef')) .or. &
            (codepoint >= int(z'0c66') .and. codepoint <= int(z'0c6f')) .or. &
            (codepoint >= int(z'0ce6') .and. codepoint <= int(z'0cef')) .or. &
            (codepoint >= int(z'0d66') .and. codepoint <= int(z'0d6f')) .or. &
            (codepoint >= int(z'0e50') .and. codepoint <= int(z'0e59')) .or. &
            (codepoint >= int(z'0ed0') .and. codepoint <= int(z'0ed9')) .or. &
            (codepoint >= int(z'0f20') .and. codepoint <= int(z'0f29')) .or. &
            (codepoint >= int(z'1040') .and. codepoint <= int(z'1049')) .or. &
            (codepoint >= int(z'17e0') .and. codepoint <= int(z'17e9')) .or. &
            (codepoint >= int(z'1810') .and. codepoint <= int(z'1819')) .or. &
            (codepoint >= int(z'ff10') .and. codepoint <= int(z'ff19'))
    end function qwen35_is_number

    logical function qwen35_is_mark(codepoint)
        integer, intent(in) :: codepoint

        qwen35_is_mark = (codepoint >= int(z'0300') .and. codepoint <= int(z'036f')) .or. &
            (codepoint >= int(z'0483') .and. codepoint <= int(z'0489')) .or. &
            (codepoint >= int(z'0591') .and. codepoint <= int(z'05bd')) .or. &
            (codepoint >= int(z'0610') .and. codepoint <= int(z'061a')) .or. &
            (codepoint >= int(z'064b') .and. codepoint <= int(z'065f')) .or. &
            (codepoint >= int(z'0900') .and. codepoint <= int(z'0903')) .or. &
            (codepoint >= int(z'093a') .and. codepoint <= int(z'094f')) .or. &
            (codepoint >= int(z'0981') .and. codepoint <= int(z'0983')) .or. &
            (codepoint >= int(z'09bc') .and. codepoint <= int(z'09cd')) .or. &
            (codepoint >= int(z'0a01') .and. codepoint <= int(z'0a03')) .or. &
            (codepoint >= int(z'0abc') .and. codepoint <= int(z'0acd')) .or. &
            (codepoint >= int(z'0b01') .and. codepoint <= int(z'0b03')) .or. &
            (codepoint >= int(z'0b3c') .and. codepoint <= int(z'0bcd')) .or. &
            (codepoint >= int(z'0c00') .and. codepoint <= int(z'0c04')) .or. &
            (codepoint >= int(z'0c3e') .and. codepoint <= int(z'0c4d')) .or. &
            (codepoint >= int(z'0d3b') .and. codepoint <= int(z'0d4d')) .or. &
            (codepoint >= int(z'0e31') .and. codepoint <= int(z'0e4d')) .or. &
            (codepoint >= int(z'0f18') .and. codepoint <= int(z'0f39'))
    end function qwen35_is_mark

    logical function qwen35_is_letter(codepoint)
        integer, intent(in) :: codepoint

        qwen35_is_letter = (codepoint >= int(z'41') .and. codepoint <= int(z'5a')) .or. &
            (codepoint >= int(z'61') .and. codepoint <= int(z'7a')) .or. &
            (codepoint >= int(z'00c0') .and. codepoint <= int(z'02af')) .or. &
            (codepoint >= int(z'0370') .and. codepoint <= int(z'052f')) .or. &
            (codepoint >= int(z'0531') .and. codepoint <= int(z'058f')) .or. &
            (codepoint >= int(z'0590') .and. codepoint <= int(z'06ff')) .or. &
            (codepoint >= int(z'0700') .and. codepoint <= int(z'074f')) .or. &
            (codepoint >= int(z'0780') .and. codepoint <= int(z'07bf')) .or. &
            (codepoint >= int(z'0900') .and. codepoint <= int(z'0d7f')) .or. &
            (codepoint >= int(z'0e80') .and. codepoint <= int(z'0eff')) .or. &
            (codepoint >= int(z'1000') .and. codepoint <= int(z'1fff')) .or. &
            (codepoint >= int(z'2e80') .and. codepoint <= int(z'9fff')) .or. &
            (codepoint >= int(z'ac00') .and. codepoint <= int(z'd7af')) .or. &
            (codepoint >= int(z'f900') .and. codepoint <= int(z'faff')) .or. &
            (codepoint >= int(z'ff21') .and. codepoint <= int(z'ff5a')) .or. &
            (codepoint >= int(z'20000') .and. codepoint <= int(z'2ffff'))
    end function qwen35_is_letter

    logical function qwen35_is_word(codepoint)
        integer, intent(in) :: codepoint

        qwen35_is_word = .false.
        if (qwen35_is_number(codepoint)) return
        qwen35_is_word = qwen35_is_letter(codepoint) .or. qwen35_is_mark(codepoint)
    end function qwen35_is_word

    logical function qwen35_is_newline(codepoint)
        integer, intent(in) :: codepoint
        qwen35_is_newline = codepoint == 10 .or. codepoint == 13
    end function qwen35_is_newline

    logical function qwen35_is_whitespace(codepoint)
        integer, intent(in) :: codepoint

        qwen35_is_whitespace = codepoint == 9 .or. codepoint == 10 .or. codepoint == 11 .or. &
            codepoint == 12 .or. codepoint == 13 .or. codepoint == 32 .or. &
            codepoint == int(z'0085') .or. codepoint == int(z'00a0') .or. &
            (codepoint >= int(z'2000') .and. codepoint <= int(z'200a')) .or. &
            codepoint == int(z'2028') .or. codepoint == int(z'2029') .or. codepoint == int(z'202f') .or. &
            codepoint == int(z'205f') .or. codepoint == int(z'3000')
    end function qwen35_is_whitespace

    logical function qwen35_is_optional_prefix(codepoint)
        integer, intent(in) :: codepoint

        qwen35_is_optional_prefix = .not. qwen35_is_newline(codepoint) .and. &
            .not. qwen35_is_number(codepoint) .and. .not. qwen35_is_word(codepoint)
    end function qwen35_is_optional_prefix

    subroutine native_tokenizer_encode(self, text, ids, add_special, parse_special)
        class(fortai_native_tokenizer_t), intent(in) :: self
        character(len=*), intent(in) :: text
        integer(int32), allocatable, intent(out) :: ids(:)
        logical, intent(in), optional :: add_special, parse_special
        integer(int32), allocatable :: work(:)
        integer :: capacity, count, start, finish, relative, special_end, id
        integer :: marker_length, candidate, marker_position
        character(len=:), allocatable :: special
        logical :: add_special_value, parse_special_value
        character(len=16), parameter :: inline_specials(6) = [character(len=16) :: &
            '<think>', '</think>', '<tool_call>', '</tool_call>', '<tool_response>', '</tool_response>']

        capacity = max(32, 4 * len(text) + 8)
        allocate(work(capacity))
        count = 0
        add_special_value = self%add_bos_token
        if (present(add_special)) add_special_value = add_special
        parse_special_value = .true.
        if (present(parse_special)) parse_special_value = parse_special
        if (add_special_value .and. self%bos_token >= 0) then
            count = 1
            work(count) = self%bos_token
        end if
        if (.not. parse_special_value) then
            call encode_plain(self, text, work, count, capacity)
            if (allocated(ids)) deallocate(ids)
            allocate(ids(count))
            if (count > 0) ids = work(:count)
            deallocate(work)
            return
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

    subroutine native_tokenizer_token_piece(self, token, piece, valid)
        class(fortai_native_tokenizer_t), intent(in) :: self
        integer(int32), intent(in) :: token
        character(len=:), allocatable, intent(out) :: piece
        logical, intent(out) :: valid
        integer :: index, length, position, codepoint, width, total, byte_value, i
        character(len=:), allocatable :: raw, encoded

        valid = .false.
        allocate(character(len=0) :: piece)
        if (token < 0_int32 .or. token >= self%vocab_size) return
        if (.not. allocated(self%vocabulary_offset) .or. .not. allocated(self%vocabulary_length)) return
        index = int(token) + 1
        if (index < 1 .or. index > size(self%vocabulary_offset)) return
        length = int(self%vocabulary_length(index))
        if (length <= 0) then
            valid = .true.
            return
        end if
        raw = self%vocabulary_blob(self%vocabulary_offset(index): &
            self%vocabulary_offset(index) + length - 1)
        ! GGUF BPE vocabularies store arbitrary bytes through the GPT-2
        ! byte-to-unicode alphabet.  The HTTP piece API, like llama.cpp's
        ! common_token_to_piece(), must expose those original bytes rather
        ! than the printable surrogate code points (for example, emoji
        ! fragments are returned as a JSON byte array when not valid UTF-8).
        total = 0
        position = 1
        do while (position <= len(raw))
            call utf8_decode(raw, position, codepoint, width)
            if (codepoint >= 256 .and. codepoint < 512) then
                total = total + 1
            else if (codepoint <= 255) then
                total = total + 1
            else
                encoded = utf8_encode(codepoint)
                total = total + len(encoded)
            end if
            position = position + width
        end do
        deallocate(piece)
        allocate(character(len=total) :: piece)
        total = 0
        position = 1
        do while (position <= len(raw))
            call utf8_decode(raw, position, codepoint, width)
            if (codepoint >= 256 .and. codepoint < 512) then
                byte_value = unicode_byte(codepoint)
                total = total + 1
                piece(total:total) = achar(byte_value)
            else if (codepoint <= 255) then
                byte_value = codepoint
                total = total + 1
                piece(total:total) = achar(byte_value)
            else
                encoded = utf8_encode(codepoint)
                do i = 1, len(encoded)
                    total = total + 1
                    piece(total:total) = encoded(i:i)
                end do
            end if
            position = position + width
        end do
        valid = .true.
    end subroutine native_tokenizer_token_piece

    subroutine native_tokenizer_decode(self, ids, text)
        class(fortai_native_tokenizer_t), intent(in) :: self
        integer(int32), intent(in) :: ids(:)
        character(len=:), allocatable, intent(out) :: text
        integer :: i, j, k, codepoint, width, byte_value, total, output_total
        integer :: valid_width, output_position
        character(len=:), allocatable :: piece
        character(len=:), allocatable :: raw, encoded
        total = 0
        do i = 1, size(ids)
            if (ids(i) < 0 .or. ids(i) >= self%vocab_size) cycle
            if (self%vocabulary_length(ids(i) + 1) == 0) then
                piece = ''
            else
                piece = self%vocabulary_blob(self%vocabulary_offset(ids(i) + 1): &
                    self%vocabulary_offset(ids(i) + 1) + self%vocabulary_length(ids(i) + 1) - 1)
            end if
            j = 1
            do while (j <= len(piece))
                call utf8_decode(piece, j, codepoint, width)
                total = total + decoded_codepoint_width(codepoint)
                j = j + width
            end do
        end do
        allocate(character(len=total) :: raw)
        total = 0
        do i = 1, size(ids)
            if (ids(i) < 0 .or. ids(i) >= self%vocab_size) cycle
            if (self%vocabulary_length(ids(i) + 1) == 0) then
                piece = ''
            else
                piece = self%vocabulary_blob(self%vocabulary_offset(ids(i) + 1): &
                    self%vocabulary_offset(ids(i) + 1) + self%vocabulary_length(ids(i) + 1) - 1)
            end if
            j = 1
            do while (j <= len(piece))
                call utf8_decode(piece, j, codepoint, width)
                if (codepoint >= 256 .and. codepoint < 512) then
                    byte_value = unicode_byte(codepoint)
                    total = total + 1
                    raw(total:total) = achar(byte_value)
                else if (codepoint <= 255) then
                    byte_value = codepoint
                    total = total + 1
                    raw(total:total) = achar(byte_value)
                else
                    encoded = utf8_encode(codepoint)
                    do k = 1, len(encoded)
                        total = total + 1
                        raw(total:total) = encoded(k:k)
                    end do
                end if
                j = j + width
            end do
        end do
        output_total = 0
        j = 1
        do while (j <= total)
            valid_width = valid_utf8_width(raw, j)
            if (valid_width > 0) then
                output_total = output_total + valid_width
                j = j + valid_width
            else
                output_total = output_total + 3
                j = j + 1
            end if
        end do
        allocate(character(len=output_total) :: text)
        output_position = 1
        j = 1
        do while (j <= total)
            valid_width = valid_utf8_width(raw, j)
            if (valid_width > 0) then
                text(output_position:output_position + valid_width - 1) = raw(j:j + valid_width - 1)
                output_position = output_position + valid_width
                j = j + valid_width
            else
                text(output_position:output_position + 2) = char(int(z'ef')) // char(int(z'bf')) // char(int(z'bd'))
                output_position = output_position + 3
                j = j + 1
            end if
        end do
    end subroutine native_tokenizer_decode

    integer function decoded_codepoint_width(codepoint)
        integer, intent(in) :: codepoint

        if (codepoint <= 511) then
            decoded_codepoint_width = 1
        else if (codepoint <= int(z'7ff')) then
            decoded_codepoint_width = 2
        else if (codepoint <= int(z'ffff')) then
            decoded_codepoint_width = 3
        else
            decoded_codepoint_width = 4
        end if
    end function decoded_codepoint_width

    integer function valid_utf8_width(text, position)
        character(len=*), intent(in) :: text
        integer, intent(in) :: position
        integer :: first, second, third, fourth, codepoint

        valid_utf8_width = 0
        if (position > len(text)) return
        first = iachar(text(position:position))
        if (first < int(z'80')) then
            valid_utf8_width = 1
            return
        end if
        if (first >= int(z'c2') .and. first <= int(z'df')) then
            if (position + 1 > len(text)) return
            second = iachar(text(position + 1:position + 1))
            if (second >= int(z'80') .and. second <= int(z'bf')) valid_utf8_width = 2
            return
        end if
        if (first >= int(z'e0') .and. first <= int(z'ef')) then
            if (position + 2 > len(text)) return
            second = iachar(text(position + 1:position + 1))
            third = iachar(text(position + 2:position + 2))
            if (second < int(z'80') .or. second > int(z'bf')) return
            if (third < int(z'80') .or. third > int(z'bf')) return
            codepoint = iand(first, 15) * 4096 + iand(second, 63) * 64 + iand(third, 63)
            if (codepoint < int(z'800') .or. (codepoint >= int(z'd800') .and. codepoint <= int(z'dfff'))) return
            valid_utf8_width = 3
            return
        end if
        if (first >= int(z'f0') .and. first <= int(z'f4')) then
            if (position + 3 > len(text)) return
            second = iachar(text(position + 1:position + 1))
            third = iachar(text(position + 2:position + 2))
            fourth = iachar(text(position + 3:position + 3))
            if (second < int(z'80') .or. second > int(z'bf')) return
            if (third < int(z'80') .or. third > int(z'bf')) return
            if (fourth < int(z'80') .or. fourth > int(z'bf')) return
            codepoint = iand(first, 7) * 262144 + iand(second, 63) * 4096 + &
                iand(third, 63) * 64 + iand(fourth, 63)
            if (codepoint < int(z'10000') .or. codepoint > int(z'10ffff')) return
            valid_utf8_width = 4
        end if
    end function valid_utf8_width

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
        if (index < 1 .or. .not. allocated(self%vocabulary_offset)) return
        if (index > size(self%vocabulary_offset)) return
        if (self%vocabulary_length(index) == len('<|im_end|>')) then
            native_tokenizer_is_stop = self%vocabulary_blob(self%vocabulary_offset(index): &
                self%vocabulary_offset(index) + self%vocabulary_length(index) - 1) == '<|im_end|>'
        end if
        if (.not. native_tokenizer_is_stop .and. self%vocabulary_length(index) == len('<|endoftext|>')) then
            native_tokenizer_is_stop = self%vocabulary_blob(self%vocabulary_offset(index): &
                self%vocabulary_offset(index) + self%vocabulary_length(index) - 1) == '<|endoftext|>'
        end if
    end function native_tokenizer_is_stop

end module fortai_native_tokenizer
