# ==============================================================================
# USER
# ==============================================================================

INSERT_USER = """
INSERT INTO users (
    sec_uid,
    name,
    translated_name,
    status,
    topic,
    niche,
    sub_niche,
    micro_niche,
    note,
    last_fetched,
    system_id
)
VALUES (
    :sec_uid,
    :name,
    :translated_name,
    :status,
    :topic,
    :niche,
    :sub_niche,
    :micro_niche,
    :note,
    :last_fetched,
    :system_id
)
"""

UPSERT_USER = """
INSERT INTO users (
    sec_uid,
    name,
    translated_name,
    status,
    topic,
    niche,
    sub_niche,
    micro_niche,
    note,
    last_fetched,
    system_id
)
VALUES (
    :sec_uid,
    :name,
    :translated_name,
    :status,
    :topic,
    :niche,
    :sub_niche,
    :micro_niche,
    :note,
    :last_fetched,
    :system_id
)
ON CONFLICT (sec_uid)
DO UPDATE SET
    name = EXCLUDED.name,
    translated_name = EXCLUDED.translated_name,
    status = EXCLUDED.status,
    topic = EXCLUDED.topic,
    niche = EXCLUDED.niche,
    sub_niche = EXCLUDED.sub_niche,
    micro_niche = EXCLUDED.micro_niche,
    note = EXCLUDED.note,
    last_fetched = EXCLUDED.last_fetched,
    system_id = EXCLUDED.system_id
"""

SELECT_USERS = "SELECT * FROM users"

SELECT_USERS_TO_FETCH = """
SELECT *
FROM users
WHERE (
    status = 'active'
    AND (
        last_fetched IS NULL
        OR last_fetched < unixepoch('now', 'start of day')
    )
)
OR (
    status = 'testing'
    AND last_fetched IS NULL
)
"""

SELECT_USER_BY_ID = "SELECT * FROM users WHERE id = :id"

SELECT_USER_BY_SEC_UID = "SELECT * FROM users WHERE sec_uid = :sec_uid"

UPDATE_USER_BY_ID = """
UPDATE users
SET
    name = :name,
    translated_name = :translated_name,
    status = :status,
    topic = :topic,
    niche = :niche,
    sub_niche = :sub_niche,
    micro_niche = :micro_niche,
    note = :note,
    last_fetched = :last_fetched,
    system_id = :system_id
WHERE id = :id
"""

DELETE_USER_BY_ID = "DELETE FROM users WHERE id = :id"


# ==============================================================================
# VIDEO
# ==============================================================================

INSERT_VIDEO = """
INSERT INTO videos (
    aweme_id,
    title,
    translated_title,
    create_time,
    digg_count,
    duration,
    urls,
    is_downloaded,
    user_id
)
VALUES (
    :aweme_id,
    :title,
    :translated_title,
    :create_time,
    :digg_count,
    :duration,
    :urls,
    :is_downloaded,
    :user_id
)
"""

UPSERT_VIDEO = """
INSERT INTO videos (
    aweme_id,
    title,
    translated_title,
    create_time,
    digg_count,
    duration,
    urls,
    is_downloaded,
    user_id
)
VALUES (
    :aweme_id,
    :title,
    :translated_title,
    :create_time,
    :digg_count,
    :duration,
    :urls,
    :is_downloaded,
    :user_id
)
ON CONFLICT (aweme_id)
DO UPDATE SET
    title = EXCLUDED.title,
    translated_title = EXCLUDED.translated_title,
    create_time = EXCLUDED.create_time,
    digg_count = EXCLUDED.digg_count,
    duration = EXCLUDED.duration,
    urls = EXCLUDED.urls,
    user_id = EXCLUDED.user_id
"""

SELECT_VIDEOS = "SELECT * FROM videos"

COUNT_VIDEOS = "SELECT COUNT(*) AS total FROM videos"

SELECT_VIDEOS_PAGE = """
SELECT *
FROM videos
ORDER BY create_time DESC, id DESC
LIMIT :limit OFFSET :offset
"""

SELECT_VIDEOS_TO_DOWNLOAD = """
SELECT v.* FROM videos v
INNER JOIN users u ON v.user_id = u.id
WHERE u.status IN ('active', 'testing')
AND v.is_downloaded = 0
"""
# OR date(last_fetched, 'unixepoch') < date('now')

SELECT_VIDEOS_TO_DOWNLOAD_BY_USER_IDS = """
SELECT v.* FROM videos v
WHERE v.is_downloaded = 0
AND v.user_id IN ({placeholders})
"""


SELECT_VIDEO_BY_ID = "SELECT * FROM videos WHERE id = :id"

SELECT_VIDEO_BY_AWEME_ID = "SELECT * FROM videos WHERE aweme_id = :aweme_id"

SELECT_VIDEOS_BY_USER_ID = "SELECT * FROM videos WHERE user_id = :user_id"

SELECT_SYSTEM_NAME_BY_VIDEO_ID = """
SELECT s.name, u.translated_name
FROM videos v
INNER JOIN users u ON v.user_id = u.id
LEFT JOIN systems s ON u.system_id = s.id
WHERE v.id = :id
"""

UPDATE_VIDEO_BY_ID = """
UPDATE videos
SET
    title = :title,
    translated_title = :translated_title,
    digg_count = :digg_count,
    duration = :duration,
    urls = :urls,
    is_downloaded = :is_downloaded
WHERE id = :id
"""

DELETE_VIDEO_BY_ID = "DELETE FROM videos WHERE id = :id"
