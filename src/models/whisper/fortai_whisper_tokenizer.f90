module fortai_whisper_tokenizer
    !! Whisper vocabulary helpers.
    !!
    !! Legacy Whisper files store the exact byte strings used by whisper.cpp.
    !! Pieces must therefore be copied byte-for-byte: ASCII spaces are real
    !! bytes, while multilingual pieces may be UTF-8 or intentionally invalid
    !! byte sequences from the GPT-2 alphabet.
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

        call piece%clear()
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
        call piece%append(raw)
        if (present(valid)) valid = .true.
    end subroutine whisper_token_piece

end module fortai_whisper_tokenizer
