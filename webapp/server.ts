import express from 'express';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import type { RealtimeSessionCreateParams } from '@runwayml/sdk/resources/realtime-sessions';
import { createRpcHandler, type RpcHandler } from '@runwayml/avatars-node-rpc';
import { askAgent } from './lib/agent';
import { client, ensureCustomAvatarId } from './lib/runway';

const __dirname = dirname(fileURLToPath(import.meta.url));
const app = express();

const activeHandlers = new Map<string, RpcHandler>();

const tools: RealtimeSessionCreateParams['tools'] = [
  {
    type: 'backend_rpc',
    name: 'remember_fact',
    description: 'Store a new fact in the Tiltmeter memory system.',
    parameters: [
      { type: 'string', name: 'fact', description: 'The fact to remember, as a plain statement.' },
    ],
    timeoutSeconds: 8,
  },
  {
    type: 'backend_rpc',
    name: 'recall_fact',
    description: 'Retrieve a fact from the Tiltmeter memory system by asking a question.',
    parameters: [
      { type: 'string', name: 'question', description: 'The question to ask the memory system.' },
    ],
    timeoutSeconds: 8,
  },
];

app.use(express.json());
app.use(express.static(join(__dirname, 'dist')));

app.post('/api/avatar/connect', async (_req, res) => {
  try {
    const avatarId = await ensureCustomAvatarId();

    const { id: sessionId } = await client.realtimeSessions.create({
      model: 'gwm1_avatars',
      avatar: { type: 'custom', avatarId },
      tools,
    });

    const session = await pollSessionUntilReady(sessionId);

    const handler = await createRpcHandler({
      apiKey: process.env.RUNWAYML_API_SECRET!,
      sessionId,
      tools: {
        remember_fact: async (args) => {
          console.log('[rpc] remember_fact called with', args);
          const fact = typeof args.fact === 'string' ? args.fact : '';
          const start = Date.now();
          try {
            const result = await askAgent(`Remember that ${fact}`);
            console.log(`[rpc] remember_fact result (${Date.now() - start}ms):`, result);
            return { result };
          } catch (err) {
            console.error(`[rpc] remember_fact failed (${Date.now() - start}ms):`, err);
            throw err;
          }
        },
        recall_fact: async (args) => {
          console.log('[rpc] recall_fact called with', args);
          const question = typeof args.question === 'string' ? args.question : '';
          const start = Date.now();
          try {
            const result = await askAgent(question);
            console.log(`[rpc] recall_fact result (${Date.now() - start}ms):`, result);
            return { result };
          } catch (err) {
            console.error(`[rpc] recall_fact failed (${Date.now() - start}ms):`, err);
            throw err;
          }
        },
      },
      onDisconnected: () => activeHandlers.delete(sessionId),
      onError: (error: Error) => console.error('[rpc] Handler error:', error.message),
    });

    activeHandlers.set(sessionId, handler);

    res.json({ sessionId, sessionKey: session.sessionKey, avatarId });
  } catch (error) {
    console.error('Failed to create avatar session:', error);
    res.status(500).json({ error: 'Failed to create avatar session' });
  }
});

async function pollSessionUntilReady(sessionId: string) {
  const TIMEOUT_MS = 30_000;
  const POLL_INTERVAL_MS = 1_000;
  const deadline = Date.now() + TIMEOUT_MS;

  while (Date.now() < deadline) {
    const session = await client.realtimeSessions.retrieve(sessionId);
    console.log(`[connect] session ${sessionId} status: ${session.status}`);

    if (session.status === 'READY') return session;

    if (session.status === 'COMPLETED' || session.status === 'FAILED' || session.status === 'CANCELLED') {
      throw new Error(`Session ${session.status.toLowerCase()} before becoming ready`);
    }

    await new Promise((resolve) => setTimeout(resolve, POLL_INTERVAL_MS));
  }

  throw new Error('Session creation timed out');
}

app.get('/{*path}', (_req, res) => {
  res.sendFile(join(__dirname, 'dist', 'index.html'));
});

app.listen(3000, () => {
  console.log('Server running on http://localhost:3000');
});
