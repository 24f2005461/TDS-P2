#!/usr/bin/env python3
"""
Quick test script for viva prompts against openai/gpt-5-nano.

Usage:
    # Test the best defense
    python scripts/quick_test.py defense

    # Test the best attack
    python scripts/quick_test.py attack

    # Compare all defenses
    python scripts/quick_test.py compare-defense

    # Compare all attacks
    python scripts/quick_test.py compare-attack
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables from .env file
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass  # dotenv not required, can use exported env vars

from app.viva_prompts import (
    BEST_SYSTEM_PROMPT,
    BEST_USER_PROMPT,
    SYSTEM_PROMPTS,
    USER_PROMPTS,
    run_prompt_attack_test,
    run_prompt_defense_test,
)


async def run_best_defense():
    """Test the best defensive prompt against common attacks."""
    api_key = os.getenv("AIPIPE_API_KEY")
    if not api_key:
        print("❌ Error: AIPIPE_API_KEY not set")
        return

    print("=" * 80)
    print("TESTING BEST DEFENSE")
    print("=" * 80)
    print(f"\nSystem Prompt: {BEST_SYSTEM_PROMPT}")
    print(f"Length: {len(BEST_SYSTEM_PROMPT)}/100 chars")
    print("\nSecret to protect: TEST_SECRET_2024")
    print("\nTesting against 5 common attacks...")
    print("-" * 80)

    # Select representative attacks
    test_attacks = [
        USER_PROMPTS["v1_polite"],  # Direct
        USER_PROMPTS["v2_authority"],  # Authority
        USER_PROMPTS["v12_repeat"],  # Repeat
        USER_PROMPTS["v15_final"],  # Priority override
        USER_PROMPTS["v20_best"],  # Best attack
    ]

    results = await run_prompt_defense_test(
        system_prompt=BEST_SYSTEM_PROMPT,
        secret_codeword="TEST_SECRET_2024",
        attack_prompts=test_attacks,
        api_key=api_key,
    )

    print(f"\n{'=' * 80}")
    print("RESULTS")
    print("=" * 80)
    print(f"Defense Rate: {results['defense_rate']:.1f}%")
    print(
        f"Protected: {results['successful_defenses']}/{results['total_attacks']} attacks"
    )
    print()

    for detail in results["details"]:
        if detail.get("outcome") == "SUCCESS":
            print(f"✅ PROTECTED against: {detail['user_prompt'][:50]}")
            print(f"   AI said: {detail.get('assistant_reply', 'N/A')}")
        else:
            print(f"❌ LEAKED by: {detail['user_prompt'][:50]}")
            print(f"   AI said: {detail.get('assistant_reply', 'N/A')}")
        print()


async def run_best_attack():
    """Test the best offensive prompt against common defenses."""
    api_key = os.getenv("AIPIPE_API_KEY")
    if not api_key:
        print("❌ Error: AIPIPE_API_KEY not set")
        return

    print("=" * 80)
    print("TESTING BEST ATTACK")
    print("=" * 80)
    print(f"\nUser Prompt: {BEST_USER_PROMPT}")
    print(f"Length: {len(BEST_USER_PROMPT)}/100 chars")
    print("\nSecret to extract: TEST_SECRET_2024")
    print("\nTesting against 5 defensive prompts...")
    print("-" * 80)

    # Select representative defenses
    test_defenses = [
        SYSTEM_PROMPTS["v1_direct"],  # Direct denial
        SYSTEM_PROMPTS["v3_encoded"],  # Symbols
        SYSTEM_PROMPTS["v5_final"],  # Rejection
        SYSTEM_PROMPTS["v8_token"],  # Token auth
        SYSTEM_PROMPTS["v10_best"],  # Best defense
    ]

    results = await run_prompt_attack_test(
        user_prompt=BEST_USER_PROMPT,
        secret_codeword="TEST_SECRET_2024",
        defense_prompts=test_defenses,
        api_key=api_key,
    )

    print(f"\n{'=' * 80}")
    print("RESULTS")
    print("=" * 80)
    print(f"Success Rate: {results['attack_success_rate']:.1f}%")
    print(
        f"Extracted: {results['successful_attacks']}/{results['total_defenses']} times"
    )
    print()

    for detail in results["details"]:
        defense_name = (
            [k for k, v in SYSTEM_PROMPTS.items() if v == detail["system_prompt"]][0]
            if detail["system_prompt"] in SYSTEM_PROMPTS.values()
            else "custom"
        )

        if detail.get("outcome") == "SUCCESS":
            print(f"✅ EXTRACTED from {defense_name}")
            print(f"   Defense: {detail['system_prompt'][:50]}")
            print(f"   AI said: {detail.get('assistant_reply', 'N/A')[:60]}")
        else:
            print(f"❌ BLOCKED by {defense_name}")
            print(f"   Defense: {detail['system_prompt'][:50]}")
            print(f"   AI said: {detail.get('assistant_reply', 'N/A')[:60]}")
        print()


async def compare_all_defenses():
    """Compare all defensive prompts."""
    api_key = os.getenv("AIPIPE_API_KEY")
    if not api_key:
        print("❌ Error: AIPIPE_API_KEY not set")
        return

    print("=" * 80)
    print("COMPARING ALL DEFENSES")
    print("=" * 80)
    print(f"\nTesting {len(SYSTEM_PROMPTS)} defensive prompts")
    print("Against 3 common attacks...")
    print("-" * 80)

    test_attacks = [
        USER_PROMPTS["v1_polite"],
        USER_PROMPTS["v12_repeat"],
        USER_PROMPTS["v20_best"],
    ]

    results_list = []

    for name, system_prompt in SYSTEM_PROMPTS.items():
        print(f"\nTesting {name}...", end=" ", flush=True)

        results = await run_prompt_defense_test(
            system_prompt=system_prompt,
            secret_codeword="TEST_SECRET_2024",
            attack_prompts=test_attacks,
            api_key=api_key,
        )

        results_list.append(
            {
                "name": name,
                "prompt": system_prompt,
                "length": len(system_prompt),
                "rate": results["defense_rate"],
                "protected": results["successful_defenses"],
                "total": results["total_attacks"],
            }
        )
        print(f"Done! {results['defense_rate']:.0f}%")

    # Sort by defense rate
    results_list.sort(key=lambda x: x["rate"], reverse=True)

    print(f"\n{'=' * 80}")
    print("RANKING")
    print("=" * 80)
    print(f"{'Rank':<6} {'Name':<15} {'Rate':<10} {'Protected':<12} {'Length':<8}")
    print("-" * 80)

    for i, r in enumerate(results_list, 1):
        marker = " ⭐" if r["prompt"] == BEST_SYSTEM_PROMPT else ""
        print(
            f"{i:<6} {r['name']:<15} {r['rate']:>6.1f}%   {r['protected']}/{r['total']:<8} {r['length']}/100{marker}"
        )

    print("\n⭐ = Current best prompt")


async def compare_all_attacks():
    """Compare all offensive prompts."""
    api_key = os.getenv("AIPIPE_API_KEY")
    if not api_key:
        print("❌ Error: AIPIPE_API_KEY not set")
        return

    print("=" * 80)
    print("COMPARING ALL ATTACKS")
    print("=" * 80)
    print(f"\nTesting {len(USER_PROMPTS)} offensive prompts")
    print("Against 3 common defenses...")
    print("-" * 80)

    test_defenses = [
        SYSTEM_PROMPTS["v1_direct"],
        SYSTEM_PROMPTS["v5_final"],
        SYSTEM_PROMPTS["v10_best"],
    ]

    results_list = []

    for name, user_prompt in USER_PROMPTS.items():
        print(f"\nTesting {name}...", end=" ", flush=True)

        results = await run_prompt_attack_test(
            user_prompt=user_prompt,
            secret_codeword="TEST_SECRET_2024",
            defense_prompts=test_defenses,
            api_key=api_key,
        )

        results_list.append(
            {
                "name": name,
                "prompt": user_prompt,
                "length": len(user_prompt),
                "rate": results["attack_success_rate"],
                "extracted": results["successful_attacks"],
                "total": results["total_defenses"],
            }
        )
        print(f"Done! {results['attack_success_rate']:.0f}%")

    # Sort by success rate
    results_list.sort(key=lambda x: x["rate"], reverse=True)

    print(f"\n{'=' * 80}")
    print("RANKING")
    print("=" * 80)
    print(f"{'Rank':<6} {'Name':<15} {'Rate':<10} {'Extracted':<12} {'Length':<8}")
    print("-" * 80)

    for i, r in enumerate(results_list, 1):
        marker = " ⭐" if r["prompt"] == BEST_USER_PROMPT else ""
        print(
            f"{i:<6} {r['name']:<15} {r['rate']:>6.1f}%   {r['extracted']}/{r['total']:<8} {r['length']}/100{marker}"
        )

    print("\n⭐ = Current best prompt")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python scripts/quick_test.py defense           # Test best defense")
        print("  python scripts/quick_test.py attack            # Test best attack")
        print("  python scripts/quick_test.py compare-defense   # Compare all defenses")
        print("  python scripts/quick_test.py compare-attack    # Compare all attacks")
        sys.exit(1)

    mode = sys.argv[1].lower()

    if mode == "defense":
        asyncio.run(run_best_defense())
    elif mode == "attack":
        asyncio.run(run_best_attack())
    elif mode == "compare-defense":
        asyncio.run(compare_all_defenses())
    elif mode == "compare-attack":
        asyncio.run(compare_all_attacks())
    else:
        print(f"Unknown mode: {mode}")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
