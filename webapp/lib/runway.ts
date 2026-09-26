import Runway from '@runwayml/sdk';
import { TILTMETER_PERSONALITY, START_SCRIPT } from './personality';

export const client = new Runway({ apiKey: process.env.RUNWAYML_API_SECRET });

const MARCO_IMAGE_URL =
  'https://runway-static-assets.s3.us-east-1.amazonaws.com/calliope-demo/presets-3-3/Dev-Avatar-4.png';

let cachedAvatarId: string | null = null;

// Runway rejects `personality`/`startScript` session overrides on preset
// avatars (400: "not supported for preset avatars") — only custom avatars
// accept a system prompt, so the Tiltmeter personality is baked into a
// custom avatar built from the same reference image as the cooking-teacher
// preset, instead of overriding the preset at session-creation time.
export async function ensureCustomAvatarId(): Promise<string> {
  if (cachedAvatarId) return cachedAvatarId;

  const avatar = await client.avatars.create({
    name: 'Marco',
    personality: TILTMETER_PERSONALITY,
    startScript: START_SCRIPT,
    referenceImage: MARCO_IMAGE_URL,
    voice: { type: 'runway-live-preset', presetId: 'marcus' },
  });

  cachedAvatarId = await pollAvatarUntilReady(avatar.id);
  return cachedAvatarId;
}

async function pollAvatarUntilReady(avatarId: string): Promise<string> {
  const TIMEOUT_MS = 60_000;
  const POLL_INTERVAL_MS = 1_000;
  const deadline = Date.now() + TIMEOUT_MS;

  while (Date.now() < deadline) {
    const avatar = await client.avatars.retrieve(avatarId);

    if (avatar.status === 'READY') return avatar.id;
    if (avatar.status === 'FAILED') {
      throw new Error(`Avatar creation failed: ${avatar.failureReason}`);
    }

    await new Promise((resolve) => setTimeout(resolve, POLL_INTERVAL_MS));
  }

  throw new Error('Avatar creation timed out');
}
