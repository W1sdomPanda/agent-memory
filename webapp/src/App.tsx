import { Suspense, useCallback, useState } from 'react';
import { AvatarCall, AvatarVideo, ControlBar } from '@runwayml/avatars-react';
import '@runwayml/avatars-react/styles.css';

const AVATAR_NAME = 'Marco';

type SessionInfo = {
  sessionId: string;
  sessionKey: string;
  avatarId: string;
};

export function App() {
  const [session, setSession] = useState<SessionInfo | null>(null);
  const [isConnecting, setIsConnecting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const startCall = useCallback(async () => {
    setIsConnecting(true);
    setError(null);
    try {
      const res = await fetch('/api/avatar/connect', { method: 'POST' });
      if (!res.ok) throw new Error(`Failed to create session (${res.status})`);
      setSession(await res.json());
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to start call');
    } finally {
      setIsConnecting(false);
    }
  }, []);

  const endCall = useCallback(() => setSession(null), []);
  const onAvatarError = useCallback((err: Error) => {
    console.error(err);
    setError(err.message);
  }, []);

  return (
    <main className="page">
      <header className="header">
        <h1 className="title">Tiltmeter Memory Assistant</h1>
        <p className="description">
          Talk to {AVATAR_NAME} to remember or recall facts from Tiltmeter's
          agent memory (Qdrant + Neo4j via <code>hybrid_find</code>).
        </p>
      </header>

      {!session ? (
        <button className="start-button" onClick={startCall} disabled={isConnecting}>
          {isConnecting ? 'Connecting…' : 'Start conversation'}
        </button>
      ) : (
        <Suspense fallback={<div className="loading">Connecting…</div>}>
          <AvatarCall
            avatarId={session.avatarId}
            sessionId={session.sessionId}
            sessionKey={session.sessionKey}
            video={false}
            onEnd={endCall}
            onError={onAvatarError}
            className="call"
          >
            <AvatarVideo className="avatar-video" />
            <ControlBar showCamera={false} />
          </AvatarCall>
        </Suspense>
      )}

      {error ? <p className="error">{error}</p> : null}
    </main>
  );
}
