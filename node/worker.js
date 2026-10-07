/**
 * G.I.D.E.O.N Remote Reminder Node
 *
 * The smallest possible always-on half of G.I.D.E.O.N: it holds upcoming
 * reminders and sends them on Telegram at the right moment, so a reminder set
 * on the desktop still arrives when the desktop is shut down.
 *
 * It deliberately does NOT:
 *   - poll Telegram (getUpdates allows exactly one consumer per token, and
 *     that consumer is gideon/interfaces/telegram.py on the desktop; a second one would
 *     take the desktop bot offline with 409 Conflict)
 *   - run a model, hold conversations, or store anything but reminder text
 *   - accept anything without the shared bearer token
 *
 * Delivery ownership is exclusive. A reminder this node has accepted is fired
 * by this node; the desktop fires only the rows it failed to mirror. Nothing
 * is sent twice, and there is no de-duplication to get wrong.
 */

const JSON_HEADERS = { "content-type": "application/json" };

function json(body, status = 200) {
  return new Response(JSON.stringify(body), { status, headers: JSON_HEADERS });
}

/**
 * Constant-time-ish bearer check.
 *
 * Compares over the full length of both strings so a wrong token cannot be
 * narrowed down by timing the response.
 */
function authorized(request, env) {
  const header = request.headers.get("authorization") || "";
  const supplied = header.startsWith("Bearer ") ? header.slice(7) : "";
  const expected = env.GIDEON_NODE_TOKEN || "";
  if (!expected || supplied.length !== expected.length) return false;
  let diff = 0;
  for (let i = 0; i < expected.length; i++) {
    diff |= supplied.charCodeAt(i) ^ expected.charCodeAt(i);
  }
  return diff === 0;
}

async function sendTelegram(env, chatId, text) {
  const response = await fetch(
    `https://api.telegram.org/bot${env.TELEGRAM_BOT_TOKEN}/sendMessage`,
    {
      method: "POST",
      headers: JSON_HEADERS,
      body: JSON.stringify({ chat_id: chatId, text }),
    }
  );
  if (!response.ok) {
    console.error("Telegram send failed", response.status, await response.text());
  }
  return response.ok;
}

/**
 * Every authorised user, so a reminder created at the desktop — which has no
 * originating chat — still reaches the phone.
 */
function fallbackChats(env) {
  return (env.TELEGRAM_ALLOWED_USERS || "")
    .split(",")
    .map((value) => value.trim())
    .filter(Boolean);
}

// ---------------------------------------------------------------------------
//  HTTP API (called by remote_scheduler.py on the desktop)
// ---------------------------------------------------------------------------
export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    if (url.pathname === "/health" && request.method === "GET") {
      if (!authorized(request, env)) return json({ error: "unauthorized" }, 401);
      const row = await env.DB.prepare(
        "SELECT COUNT(*) AS pending FROM reminders WHERE status = 'pending'"
      ).first();
      return json({ status: "ok", pending: row ? row.pending : 0 });
    }

    const match = url.pathname.match(/^\/reminders\/(\d+)$/);
    if (!match) return json({ error: "not found" }, 404);
    if (!authorized(request, env)) return json({ error: "unauthorized" }, 401);

    const id = Number(match[1]);

    if (request.method === "PUT") {
      let body;
      try {
        body = await request.json();
      } catch (err) {
        return json({ error: "invalid json" }, 400);
      }
      if (!body.text || !body.remind_at) {
        return json({ error: "text and remind_at are required" }, 400);
      }
      // Upsert, so the desktop retrying a push after a timeout it never saw
      // the answer to cannot create a duplicate row.
      await env.DB.prepare(
        `INSERT INTO reminders (id, text, remind_at, chat_id, channel, status)
         VALUES (?, ?, ?, ?, ?, 'pending')
         ON CONFLICT(id) DO UPDATE SET
           text = excluded.text,
           remind_at = excluded.remind_at,
           chat_id = excluded.chat_id,
           channel = excluded.channel,
           status = 'pending'`
      )
        .bind(id, body.text, body.remind_at, body.chat_id || null, body.channel || "web")
        .run();
      return json({ status: "queued", id });
    }

    if (request.method === "DELETE") {
      // Called when the desktop delivered the reminder itself, so this copy
      // must not fire later.
      //
      // `prior` is the interesting part of the response. If this node already
      // sent the reminder while the desktop was off, the desktop's own row is
      // still 'pending' and would otherwise be reported as MISSED on the next
      // boot — a false alarm for something that was delivered on time.
      const existing = await env.DB.prepare(
        "SELECT status FROM reminders WHERE id = ?"
      )
        .bind(id)
        .first();
      await env.DB.prepare("DELETE FROM reminders WHERE id = ?").bind(id).run();
      return json({
        status: "cancelled",
        id,
        prior: existing ? existing.status : null,
      });
    }

    return json({ error: "method not allowed" }, 405);
  },

  // -------------------------------------------------------------------------
  //  Cron: fire whatever is due
  // -------------------------------------------------------------------------
  async scheduled(event, env, ctx) {
    ctx.waitUntil(fireDue(env));
  },
};

export async function fireDue(env) {
  // The desktop writes naive local-time ISO strings (datetime.isoformat), and
  // the node must compare in the same frame. GIDEON_TZ_OFFSET_MINUTES is the
  // desktop's offset from UTC — 330 for IST. Comparing against UTC instead
  // would fire every reminder hours early or late.
  const offsetMinutes = Number(env.GIDEON_TZ_OFFSET_MINUTES || 0);
  const localNow = new Date(Date.now() + offsetMinutes * 60_000);
  const cutoff = localNow.toISOString().slice(0, 19);

  const { results } = await env.DB.prepare(
    `SELECT id, text, remind_at, chat_id FROM reminders
     WHERE status = 'pending' AND remind_at <= ?
     ORDER BY remind_at ASC LIMIT 25`
  )
    .bind(cutoff)
    .all();

  if (!results || results.length === 0) return;

  for (const row of results) {
    const message = `⏰ Reminder: ${row.text}`;
    const targets = row.chat_id ? [row.chat_id] : fallbackChats(env);

    let sent = false;
    for (const chatId of targets) {
      if (await sendTelegram(env, chatId, message)) sent = true;
    }

    // Mark before retrying anything: a reminder delivered twice is worse than
    // one delivered once and logged as failed.
    await env.DB.prepare("UPDATE reminders SET status = ? WHERE id = ?")
      .bind(sent ? "sent" : "failed", row.id)
      .run();
  }

  // Keep the table from growing without bound. Nothing here is worth
  // retaining once it has been delivered.
  await env.DB.prepare(
    "DELETE FROM reminders WHERE status != 'pending' AND remind_at < ?"
  )
    .bind(new Date(localNow.getTime() - 7 * 86_400_000).toISOString().slice(0, 19))
    .run();
}
