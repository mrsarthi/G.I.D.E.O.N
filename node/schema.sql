-- G.I.D.E.O.N remote reminder node (Cloudflare D1)
--
-- `id` mirrors reminders.reminder_id on the desktop, which is what makes the
-- PUT an upsert and the DELETE addressable. Nothing about the conversation
-- that produced the reminder is stored here.

CREATE TABLE IF NOT EXISTS reminders (
    id        INTEGER PRIMARY KEY,
    text      TEXT NOT NULL,
    remind_at TEXT NOT NULL,   -- desktop-local naive ISO, e.g. 2026-08-11T09:00:00
    chat_id   TEXT,            -- NULL means broadcast to TELEGRAM_ALLOWED_USERS
    channel   TEXT DEFAULT 'web',
    status    TEXT DEFAULT 'pending'  -- pending | sent | failed
);

-- The cron query is exactly (status, remind_at).
CREATE INDEX IF NOT EXISTS idx_reminders_due
    ON reminders(status, remind_at);
