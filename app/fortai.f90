program fortai_cli
    use fortai, only: fortai_version
    implicit none

    character(len=64) :: argument
    integer :: status

    argument = ''
    call get_command_argument(1, argument, status=status)
    if (status == 0 .or. trim(argument) == '--version') then
        print '(a)', 'FortAI ' // fortai_version()
    else
        print '(a)', 'FortAI ' // fortai_version()
        print '(a)', 'Usage: fortai --version'
    end if
end program fortai_cli
