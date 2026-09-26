const AGENT_A2A_URL = process.env.AGENT_A2A_URL ?? 'http://localhost:18080/';

type A2AResponse = {
  result?: {
    artifacts?: { parts?: { text?: string }[] }[];
  };
  error?: { message?: string };
};

export async function askAgent(question: string): Promise<string> {
  const body = {
    jsonrpc: '2.0',
    id: 'avatar',
    method: 'message/send',
    params: {
      message: {
        role: 'user',
        parts: [{ kind: 'text', text: question }],
        messageId: `avatar-${Date.now()}`,
      },
    },
  };

  const res = await fetch(AGENT_A2A_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });

  const out = (await res.json()) as A2AResponse;
  if (out.error) {
    throw new Error(`A2A error: ${out.error.message}`);
  }

  const text = (out.result?.artifacts ?? [])
    .flatMap((artifact) => artifact.parts ?? [])
    .map((part) => part.text ?? '')
    .join('');

  if (!text) {
    throw new Error('A2A response had no artifact text');
  }
  return text;
}
