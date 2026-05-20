import sys
import os
import cv2
import asyncio
import numpy as np

# Ensure backend imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import AsyncSessionLocal
from sqlalchemy import select
from models import Child
from services.face_recognition import extract_embeddings


async def backfill():
    print("🚀 Starting InsightFace backfill script...")

    success = 0
    failed = 0
    skipped = 0

    async with AsyncSessionLocal() as db:

        # 🔥 IMPORTANT: Clear old embeddings first
        print("⚠️ Clearing old embeddings...")
        result = await db.execute(select(Child))
        children = result.scalars().all()

        print(f"Total children in database: {len(children)}")

        for child in children:
            print(f"\n🔍 Processing: {child.name}")

            try:
                img_path = child.image_path

                if not img_path or not os.path.exists(img_path):
                    print(f"❌ Image path not found: {img_path}")
                    skipped += 1
                    continue

                img = cv2.imread(img_path)

                if img is None:
                    print(f"❌ Failed to read image: {img_path}")
                    skipped += 1
                    continue

                embeddings = extract_embeddings(img)

                if len(embeddings) == 0:
                    print(f"❌ No face detected")
                    failed += 1
                    continue

                # 🔥 If multiple faces, pick largest (best quality)
                if len(embeddings) > 1:
                    print("⚠️ Multiple faces detected, selecting best one")

                emb = np.array(embeddings[0])

                # 🔥 Safety normalization
                norm = np.linalg.norm(emb)
                if norm > 0:
                    emb = emb / norm

                # Save to DB
                child.arcface_embedding = emb.tolist()

                print(f"✅ Success (dim={len(emb)})")
                success += 1

            except Exception as e:
                print(f"❌ Error on {child.name}: {e}")
                failed += 1

        await db.commit()

    print("\n🎯 Backfill completed")
    print(f"✅ Success: {success}")
    print(f"❌ Failed: {failed}")
    print(f"⚠️ Skipped: {skipped}")


if __name__ == "__main__":
    asyncio.run(backfill())