module fortai_whisper_tokenizer
    !! Whisper's byte-level vocabulary helpers.
    !!
    !! Legacy Whisper files contain the GPT-2 byte vocabulary directly (there
    !! is no merge table to load).  The model runtime therefore only needs the
    !! inverse byte-to-unicode map for decoding generated pieces.
    use, intrinsic :: iso_fortran_env, only: int32
    use fortai_string, only: string_t
    use fortai_status, only: FORTAI_INVALID, status_t
    use fortai_whisper_format, only: whisper_file_t
    implicit none
    private

    integer(int32), parameter, public :: WHISPER_TOKEN_EOT = 50257_int32
    integer(int32), parameter, public :: WHISPER_TOKEN_SOT = 50258_int32
    integer(int32), parameter, public :: WHISPER_TOKEN_TRANSLATE = 50359_int32
    integer(int32), parameter, public :: WHISPER_TOKEN_TRANSCRIBE = 50360_int32
    integer(int32), parameter, public :: WHISPER_TOKEN_SOLM = 50361_int32
    integer(int32), parameter, public :: WHISPER_TOKEN_PREV = 50362_int32
    integer(int32), parameter, public :: WHISPER_TOKEN_NOSP = 50363_int32
    integer(int32), parameter, public :: WHISPER_TOKEN_NOT = 50364_int32
    integer(int32), parameter, public :: WHISPER_TOKEN_BEG = 50365_int32

    character(len=3), parameter :: LANGUAGE_CODES(100) = [ character(len=3) :: &
        'en ','zh ','de ','es ','ru ','ko ','fr ','ja ','pt ','tr ','pl ','ca ','nl ','ar ','sv ','it ', &
        'id ','hi ','fi ','vi ','he ','uk ','el ','ms ','cs ','ro ','da ','hu ','ta ','no ','th ','ur ', &
        'hr ','bg ','lt ','la ','mi ','ml ','cy ','sk ','te ','fa ','lv ','bn ','sr ','az ','sl ','kn ', &
        'et ','mk ','br ','eu ','is ','hy ','ne ','mn ','bs ','kk ','sq ','sw ','gl ','mr ','pa ','si ', &
        'km ','sn ','yo ','so ','af ','oc ','ka ','be ','tg ','sd ','gu ','am ','yi ','lo ','uz ','fo ', &
        'ht ','ps ','tk ','nn ','mt ','sa ','lb ','my ','bo ','tl ','mg ','as ','tt ','haw','ln ','ha ', &
        'ba ','jw ','su ','yue' ]

    public :: whisper_token_piece
    public :: whisper_language_token
    public :: whisper_token_is_special
    public :: whisper_token_name

contains

    integer(int32) function whisper_language_token(language, stat)
        character(len=*), intent(in) :: language
        type(status_t), intent(out), optional :: stat
        integer :: i, code
        character(len=3) :: wanted

        if (present(stat)) call stat%clear()
        whisper_language_token = -1_int32
        wanted = '   '
        if (len_trim(language) > 0) then
            wanted(1:min(3, len_trim(language))) = language(1:min(3, len_trim(language)))
            do i = 1, min(3, len_trim(language))
                code = iachar(wanted(i:i))
                if (code >= iachar('A') .and. code <= iachar('Z')) wanted(i:i) = achar(code + 32)
            end do
        end if
        do i = 1, size(LANGUAGE_CODES)
            if (wanted == LANGUAGE_CODES(i)) then
                whisper_language_token = WHISPER_TOKEN_SOT + int(i, int32)
                return
            end if
        end do
        if (present(stat)) call stat%set(FORTAI_INVALID, 'unsupported Whisper language: ' // trim(language))
    end function whisper_language_token

    logical function whisper_token_is_special(token)
        integer(int32), intent(in) :: token

        whisper_token_is_special = token >= WHISPER_TOKEN_EOT
    end function whisper_token_is_special

    function whisper_token_name(token) result(name)
        integer(int32), intent(in) :: token
        character(len=:), allocatable :: name

        select case (token)
        case (WHISPER_TOKEN_EOT); name = '<|endoftext|>'
        case (WHISPER_TOKEN_SOT); name = '<|startoftranscript|>'
        case (WHISPER_TOKEN_TRANSLATE); name = '<|translate|>'
        case (WHISPER_TOKEN_TRANSCRIBE); name = '<|transcribe|>'
        case (WHISPER_TOKEN_SOLM); name = '<|startoflm|>'
        case (WHISPER_TOKEN_PREV); name = '<|startofprev|>'
        case (WHISPER_TOKEN_NOSP); name = '<|nospeech|>'
        case (WHISPER_TOKEN_NOT); name = '<|notimestamps|>'
        case (WHISPER_TOKEN_BEG); name = '<|0.00|>'
        case default
            if (token > WHISPER_TOKEN_SOT .and. token < WHISPER_TOKEN_TRANSLATE) then
                if (token - WHISPER_TOKEN_SOT <= size(LANGUAGE_CODES)) then
                    name = '<|' // trim(LANGUAGE_CODES(token - WHISPER_TOKEN_SOT)) // '|>'
                else
                    name = ''
                end if
            else
                name = ''
            end if
        end select
    end function whisper_token_name

    subroutine whisper_token_piece(file, token, piece, valid)
        type(whisper_file_t), intent(in) :: file
        integer(int32), intent(in) :: token
        type(string_t), intent(out) :: piece
        logical, intent(out), optional :: valid
        character(len=:), allocatable :: raw
        integer :: position, codepoint, next_position, byte_value
        logical :: okay

        call piece%clear()
        okay = .false.
        if (token < 0_int32 .or. .not. allocated(file%vocab%token)) then
            if (present(valid)) valid = .false.
            return
        end if
        if (token + 1_int32 > size(file%vocab%token)) then
            if (present(valid)) valid = whisper_token_is_special(token)
            return
        end if
        raw = file%vocab%token(token + 1_int32)%as_character()
        if (len(raw) == 0) then
            if (present(valid)) valid = .true.
            return
        end if
        position = 1
        do while (position <= len(raw))
            call utf8_codepoint(raw, position, codepoint, next_position, okay)
            if (.not. okay) then
                call piece%append_char(raw(position:position))
                position = position + 1
                cycle
            end if
            byte_value = gpt2_codepoint_to_byte(codepoint)
            if (byte_value >= 0) call piece%append_char(achar(byte_value))
            position = next_position
        end do
        if (present(valid)) valid = .true.
    end subroutine whisper_token_piece

    subroutine utf8_codepoint(text, first, codepoint, next, okay)
        character(len=*), intent(in) :: text
        integer, intent(in) :: first
        integer(int32), intent(out) :: codepoint
        integer, intent(out) :: next
        logical, intent(out) :: okay
        integer :: b0, b1, b2, b3, count

        b0 = iachar(text(first:first))
        codepoint = 0_int32
        next = first + 1
        okay = .false.
        if (b0 < 128) then
            codepoint = b0
            okay = .true.
            return
        end if
        if (b0 >= 194 .and. b0 <= 223) then
            count = 2
        else if (b0 >= 224 .and. b0 <= 239) then
            count = 3
        else if (b0 >= 240 .and. b0 <= 244) then
            count = 4
        else
            return
        end if
        if (first + count - 1 > len(text)) return
        b1 = iachar(text(first + 1:first + 1))
        if (b1 < 128 .or. b1 > 191) return
        if (count == 2) then
            codepoint = int(b0 - 192, int32) * 64_int32 + int(b1 - 128, int32)
            next = first + 2
            okay = .true.
            return
        end if
        b2 = iachar(text(first + 2:first + 2))
        if (b2 < 128 .or. b2 > 191) return
        if (count == 3) then
            codepoint = int(b0 - 224, int32) * 4096_int32 + int(b1 - 128, int32) * 64_int32 + int(b2 - 128, int32)
            next = first + 3
            okay = codepoint >= 2048_int32
            return
        end if
        b3 = iachar(text(first + 3:first + 3))
        if (b3 < 128 .or. b3 > 191) return
        codepoint = int(b0 - 240, int32) * 262144_int32 + int(b1 - 128, int32) * 4096_int32 + &
            int(b2 - 128, int32) * 64_int32 + int(b3 - 128, int32)
        next = first + 4
        okay = codepoint >= 65536_int32 .and. codepoint <= int(z'10ffff')
    end subroutine utf8_codepoint

    integer function gpt2_codepoint_to_byte(codepoint)
        integer(int32), intent(in) :: codepoint
        integer :: byte_value, index
        logical :: direct

        direct = (codepoint >= 33 .and. codepoint <= 126) .or. &
            (codepoint >= 161 .and. codepoint <= 172) .or. (codepoint >= 174 .and. codepoint <= 255)
        if (direct) then
            gpt2_codepoint_to_byte = codepoint
            return
        end if
        if (codepoint < 256 .or. codepoint > 511) then
            gpt2_codepoint_to_byte = -1
            return
        end if
        index = codepoint - 256
        byte_value = 0
        do while (byte_value <= 255)
            direct = (byte_value >= 33 .and. byte_value <= 126) .or. &
                (byte_value >= 161 .and. byte_value <= 172) .or. (byte_value >= 174 .and. byte_value <= 255)
            if (.not. direct) then
                if (index == 0) then
                    gpt2_codepoint_to_byte = byte_value
                    return
                end if
                index = index - 1
            end if
            byte_value = byte_value + 1
        end do
        gpt2_codepoint_to_byte = -1
    end function gpt2_codepoint_to_byte

end module fortai_whisper_tokenizer
