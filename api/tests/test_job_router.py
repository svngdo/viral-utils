from src.main import app


def test_job_routes_are_registered():
    routes = [
        route
        for route in app.routes
        if hasattr(route, "methods") and route.path.startswith("/jobs")
    ]
    paths = {route.path for route in routes}

    assert "/jobs/douyin/fetch-latest-videos" in paths
    assert "/jobs/douyin/users/fetch-videos" in paths
    assert "/jobs/douyin/users/{user_id}/fetch-videos" in paths
    assert "/jobs/video/process-videos" in paths
    assert "/jobs/{job_id}" in paths
    assert "/jobs/{job_id}/cancel" in paths
    assert "/jobs/{job_id}/events" in paths
    assert "/jobs/fetch-latest-videos" not in paths
    assert "/jobs/process-videos" not in paths

    route_paths = [route.path for route in routes]
    assert route_paths.index("/jobs/douyin/users/fetch-videos") < route_paths.index(
        "/jobs/douyin/users/{user_id}/fetch-videos"
    )


def test_legacy_stream_cancel_route_is_not_registered():
    routes = [
        route
        for route in app.routes
        if hasattr(route, "methods") and route.path == "/video/process/stream"
    ]

    assert not any("DELETE" in route.methods for route in routes)
