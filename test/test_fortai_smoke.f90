program test_fortai_smoke
    use, intrinsic :: iso_fortran_env, only: int32, int64, real64
    use fortai_arena, only: arena_t
    use fortai_backend_cpu, only: cpu_matvec, cpu_matvec_inplace
    use fortai_cache, only: cache_path, pack_key_t
    use fortai_device, only: device_cpu, device_t
    use fortai_gguf, only: gguf_validate_header
    use fortai_model_ir, only: model_ir_t
    use fortai_plan_ir, only: plan_ir_t
    use fortai_qwen35, only: qwen35_config_t
    use fortai_qwen35_dflash2, only: qwen35_dflash2_available
    use fortai_qwen35_gdn, only: gdn_reference_step
    use fortai_qwen35_mtp, only: qwen35_mtp_available
    use fortai_qwen35_vision, only: qwen35_vision_available
    use fortai_sampler, only: sampler_t
    use fortai_speculative, only: speculative_t
    use fortai_status, only: status_t
    use fortai_tensor, only: tensor_t
    use fortai_tokenizer, only: tokenizer_t
    implicit none

    integer :: failures

    failures = 0
    call test_tensor(failures)
    call test_cpu_matvec(failures)
    call test_tokenizer(failures)
    call test_sampler(failures)
    call test_speculative(failures)
    call test_ir_contracts(failures)
    call test_device_and_arena(failures)
    call test_gguf_cache_qwen(failures)
    if (failures > 0) error stop 1
    print '(a)', 'FortAI smoke tests passed'

contains

    subroutine require(condition, message, failures)
        logical, intent(in) :: condition
        character(len=*), intent(in) :: message
        integer, intent(inout) :: failures

        if (.not. condition) then
            print '(a)', 'FAIL: ' // trim(message)
            failures = failures + 1
        end if
    end subroutine require

    subroutine test_tensor(failures)
        integer, intent(inout) :: failures
        type(tensor_t) :: tensor
        type(status_t) :: stat

        call tensor%init([2_int32, 3_int32], stat)
        call require(stat%is_ok(), 'tensor initialization', failures)
        call require(tensor%numel() == 6_int64, 'tensor element count', failures)
        call tensor%fill(2.5_real64)
        call require(abs(tensor%value_at(4_int32, stat) - 2.5_real64) < 1.0e-12_real64, &
            'tensor fill and access', failures)
        call require(stat%is_ok(), 'tensor access status', failures)
    end subroutine test_tensor

    subroutine test_cpu_matvec(failures)
        integer, intent(inout) :: failures
        real(real64) :: matrix(2, 2), vector(2)
        real(real64) :: inplace_result(2)
        real(real64), allocatable :: result(:)
        type(status_t) :: stat

        matrix = reshape([1.0_real64, 3.0_real64, 2.0_real64, 4.0_real64], [2, 2])
        vector = [5.0_real64, 6.0_real64]
        call cpu_matvec(matrix, vector, result, stat)
        call require(stat%is_ok(), 'CPU matvec status', failures)
        call require(size(result) == 2, 'CPU matvec result shape', failures)
        call require(abs(result(1) - 17.0_real64) < 1.0e-12_real64, &
            'CPU matvec first row', failures)
        call require(abs(result(2) - 39.0_real64) < 1.0e-12_real64, &
            'CPU matvec second row', failures)
        call cpu_matvec_inplace(matrix, vector, inplace_result, stat)
        call require(stat%is_ok(), 'in-place CPU matvec status', failures)
        call require(maxval(abs(inplace_result - result)) < 1.0e-12_real64, &
            'in-place CPU matvec agrees with allocating path', failures)
    end subroutine test_cpu_matvec

    subroutine test_tokenizer(failures)
        integer, intent(inout) :: failures
        type(tokenizer_t) :: tokenizer
        integer(int32), allocatable :: ids(:)
        character(len=:), allocatable :: decoded

        call tokenizer%encode('hello world', ids)
        decoded = tokenizer%decode(ids)
        call require(decoded == 'hello world', 'byte tokenizer round trip', failures)
        call require(ids(1) == 1_int32, 'tokenizer BOS token', failures)
        call require(ids(size(ids)) == 2_int32, 'tokenizer EOS token', failures)
    end subroutine test_tokenizer

    subroutine test_sampler(failures)
        integer, intent(inout) :: failures
        type(sampler_t) :: sampler
        type(status_t) :: stat
        integer(int32) :: token_id

        call sampler%greedy([0.2_real64, 1.5_real64, 1.4_real64], token_id, stat)
        call require(stat%is_ok(), 'greedy sampler status', failures)
        call require(token_id == 2_int32, 'greedy sampler argmax', failures)
    end subroutine test_sampler

    subroutine test_speculative(failures)
        integer, intent(inout) :: failures
        type(speculative_t) :: speculative
        type(status_t) :: stat
        integer(int32) :: accepted

        call speculative%accept_prefix([10_int32, 11_int32, 12_int32], &
            [10_int32, 11_int32, 99_int32], accepted, stat)
        call require(stat%is_ok(), 'speculative prefix status', failures)
        call require(accepted == 2_int32, 'speculative accepted prefix', failures)
    end subroutine test_speculative

    subroutine test_ir_contracts(failures)
        integer, intent(inout) :: failures
        type(model_ir_t) :: model
        type(plan_ir_t) :: plan
        type(status_t) :: stat

        call model%init('qwen3.5', 1024_int32, 64_int32, 2_int32, 4096_int32, &
            'Q4_K_M', stat)
        call require(stat%is_ok(), 'ModelIR validation', failures)
        call plan%init()
        call plan%append('decode', 'cpu_matvec', 1_int32, stat)
        call require(stat%is_ok(), 'PlanIR append', failures)
        call plan%validate(stat)
        call require(stat%is_ok(), 'PlanIR validation', failures)
        call require(plan%count() == 1_int32, 'PlanIR step count', failures)
    end subroutine test_ir_contracts

    subroutine test_device_and_arena(failures)
        integer, intent(inout) :: failures
        type(arena_t) :: arena
        type(device_t) :: device
        type(status_t) :: stat

        device = device_cpu()
        call require(device%is_supported(), 'CPU device support', failures)
        call require(device%label() == 'host-cpu', 'CPU device label', failures)
        call arena%init(64_int64, stat)
        call require(stat%is_ok(), 'arena initialization', failures)
        call arena%reserve(16_int64, stat)
        call require(stat%is_ok(), 'arena reservation', failures)
        call require(arena%available() == 48_int64, 'arena accounting', failures)
    end subroutine test_device_and_arena

    subroutine test_gguf_cache_qwen(failures)
        integer, intent(inout) :: failures
        real(real64) :: state(2), input(2), gate(2)
        real(real64), allocatable :: output(:)
        type(pack_key_t) :: key
        type(qwen35_config_t) :: config
        type(status_t) :: stat
        character(len=:), allocatable :: path

        call gguf_validate_header('GGUF', 3_int32, stat)
        call require(stat%is_ok(), 'GGUF header validation', failures)
        key%model = 'qwen35'
        key%backend = 'cuda'
        key%variant = 'sm86'
        path = cache_path('/tmp/fortai', key)
        call require(path == '/tmp/fortai/qwen35/cuda-sm86-abi1.pack', &
            'packed cache path', failures)
        call config%init(64_int32, 4_int32, 2_int32, 2_int32, stat)
        call require(stat%is_ok(), 'Qwen configuration validation', failures)
        state = [1.0_real64, 2.0_real64]
        input = [3.0_real64, 4.0_real64]
        gate = [1.0_real64, 2.0_real64]
        call gdn_reference_step(state, input, 0.5_real64, gate, output, stat)
        call require(stat%is_ok(), 'GDN reference status', failures)
        call require(abs(output(1) - 3.5_real64) < 1.0e-12_real64, &
            'GDN first output', failures)
        call require(abs(output(2) - 10.0_real64) < 1.0e-12_real64, &
            'GDN second output', failures)
        call require(.not. qwen35_mtp_available(), 'MTP is explicit pending work', failures)
        call require(.not. qwen35_dflash2_available(), &
            'DFlash2 is explicit pending work', failures)
        call require(.not. qwen35_vision_available(), &
            'vision is explicit pending work', failures)
    end subroutine test_gguf_cache_qwen

end program test_fortai_smoke
