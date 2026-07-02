from src.main import app


def test_job_routes_are_registered():
    paths = {
        route.path
        for route in app.routes
        if hasattr(route, "methods") and route.path.startswith("/jobs")
    }

    assert "/jobs/fetch-latest-videos" in paths
    assert "/jobs/users/{user_id}/fetch-videos" in paths
    assert "/jobs/{job_id}" in paths
    assert "/jobs/{job_id}/cancel" in paths
    assert "/jobs/{job_id}/events" in paths


def test_legacy_stream_cancel_route_is_not_registered():
    routes = [
        route
        for route in app.routes
        if hasattr(route, "methods") and route.path == "/video/process/stream"
    ]

    assert not any("DELETE" in route.methods for route in routes)
