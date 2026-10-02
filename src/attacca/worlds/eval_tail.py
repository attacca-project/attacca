POST_SUCCESS_FRAMES = 20


def post_success_tail_complete(
        success_step: int | None, current_step: int,
        post_frames: int = POST_SUCCESS_FRAMES) -> bool:
    if success_step is None:
        return False
    if int(post_frames) < 0 or int(current_step) < int(success_step):
        raise ValueError("invalid post-success tail coordinates")
    return int(current_step) - int(success_step) >= int(post_frames)


def post_success_frames_recorded(
        success_step: int | None, rows_recorded: int) -> int:
    if success_step is None:
        return 0
    return max(0, int(rows_recorded) - int(success_step) - 1)
