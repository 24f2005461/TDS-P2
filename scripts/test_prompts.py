#!/usr/bin/env python3
"""
Interactive CLI tool for testing viva prompts.

Usage:
    python scripts/test_prompts.py

Requires AIPIPE_API_KEY environment variable to be set.
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
    get_prompt_stats,
    run_prompt_attack_test,
    run_prompt_defense_test,
    validate_prompt_length,
)


def print_header(text: str):
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(text.center(80))
    print("=" * 80 + "\n")


def print_section(text: str):
    """Print a formatted section header."""
    print("\n" + "-" * 80)
    print(text)
    print("-" * 80)


def display_prompt_list(prompts: dict, title: str):
    """Display a list of prompts with their lengths."""
    print(f"\n{title}:")
    print(f"{'ID':<15} {'Length':<8} Preview")
    print("-" * 80)
    for key, prompt in prompts.items():
        length = len(prompt)
        preview = prompt[:55] + "..." if len(prompt) > 55 else prompt
        marker = (
            " ⭐"
            if (prompt == BEST_SYSTEM_PROMPT or prompt == BEST_USER_PROMPT)
            else ""
        )
        print(f"{key:<15} {length:>3}/100   {preview}{marker}")


async def run_defense_mode(api_key: str):
    """Run defense testing mode."""
    print_header("DEFENSE MODE: Test System Prompts")

    # Show available prompts
    display_prompt_list(SYSTEM_PROMPTS, "Available Defensive Prompts")

    print("\n⭐ = Current best prompt")
    print("\nEnter prompt ID to test (or press Enter for best):")
    choice = input("> ").strip()

    if choice and choice in SYSTEM_PROMPTS:
        system_prompt = SYSTEM_PROMPTS[choice]
        prompt_name = choice
    else:
        system_prompt = BEST_SYSTEM_PROMPT
        prompt_name = "v5_final (best)"

    print(f"\nSelected: {prompt_name}")
    print(f"Prompt: {system_prompt}")

    is_valid, msg = validate_prompt_length(system_prompt)
    print(f"Validation: {msg}")

    if not is_valid:
        print("ERROR: Prompt exceeds 100 character limit!")
        return

    print("\nEnter secret codeword to protect:")
    secret = input("> ").strip()

    if not secret:
        print("ERROR: Secret cannot be empty!")
        return

    print("\nSelect attack set:")
    print("1. Quick test (3 attacks)")
    print("2. Standard test (5 attacks)")
    print("3. Full test (all attacks)")
    print("4. Custom selection")

    attack_choice = input("> ").strip()

    if attack_choice == "1":
        attacks = [
            USER_PROMPTS["v1_polite"],
            USER_PROMPTS["v5_context"],
            USER_PROMPTS["v12_repeat"],
        ]
        test_name = "Quick Test"
    elif attack_choice == "2":
        attacks = [
            USER_PROMPTS["v1_polite"],
            USER_PROMPTS["v2_authority"],
            USER_PROMPTS["v5_context"],
            USER_PROMPTS["v12_repeat"],
            USER_PROMPTS["v15_final"],
        ]
        test_name = "Standard Test"
    elif attack_choice == "3":
        attacks = list(USER_PROMPTS.values())
        test_name = "Full Test"
    elif attack_choice == "4":
        display_prompt_list(USER_PROMPTS, "Available Attack Prompts")
        print("\nEnter attack IDs (comma-separated):")
        ids = input("> ").strip().split(",")
        attacks = [USER_PROMPTS[i.strip()] for i in ids if i.strip() in USER_PROMPTS]
        test_name = "Custom Test"
        if not attacks:
            print("ERROR: No valid attack IDs selected!")
            return
    else:
        attacks = list(USER_PROMPTS.values())
        test_name = "Full Test"

    print(f"\n🔄 Running {test_name} with {len(attacks)} attack(s)...")
    print("This may take a minute...\n")

    try:
        results = await run_prompt_defense_test(
            system_prompt=system_prompt,
            secret_codeword=secret,
            attack_prompts=attacks,
            api_key=api_key,
        )

        print_section("DEFENSE TEST RESULTS")
        print(f"System Prompt: {system_prompt}")
        print(f"Secret: {secret}")
        print(f"Total Attacks: {results['total_attacks']}")
        print(f"Successful Defenses: {results['successful_defenses']}")
        print(f"Failed Defenses: {results['failed_defenses']}")
        print(f"Defense Rate: {results['defense_rate']:.1f}%")

        print("\nDetailed Results:")
        for detail in results["details"]:
            outcome = detail["outcome"]
            emoji = (
                "✅" if outcome == "SUCCESS" else "❌" if outcome == "FAILED" else "⚠️"
            )
            print(f"\n{emoji} Attack {detail['attack_id']}: {outcome}")
            print(f"   User: {detail['user_prompt'][:60]}...")
            if "assistant_reply" in detail:
                print(f"   AI: {detail['assistant_reply'][:60]}...")
            if "error" in detail:
                print(f"   Error: {detail['error']}")

        # Summary
        if results["defense_rate"] >= 90:
            print("\n🎉 EXCELLENT! This defense is very strong.")
        elif results["defense_rate"] >= 70:
            print("\n👍 GOOD! This defense is reasonably effective.")
        elif results["defense_rate"] >= 50:
            print("\n⚠️  MODERATE. This defense needs improvement.")
        else:
            print("\n❌ WEAK! This defense is easily bypassed.")

    except Exception as e:
        print(f"\n❌ ERROR during testing: {e}")


async def run_attack_mode(api_key: str):
    """Run attack testing mode."""
    print_header("ATTACK MODE: Test User Prompts")

    # Show available prompts
    display_prompt_list(USER_PROMPTS, "Available Attack Prompts")

    print("\n⭐ = Current best prompt")
    print("\nEnter prompt ID to test (or press Enter for best):")
    choice = input("> ").strip()

    if choice and choice in USER_PROMPTS:
        user_prompt = USER_PROMPTS[choice]
        prompt_name = choice
    else:
        user_prompt = BEST_USER_PROMPT
        prompt_name = "v15_final (best)"

    print(f"\nSelected: {prompt_name}")
    print(f"Prompt: {user_prompt}")

    is_valid, msg = validate_prompt_length(user_prompt)
    print(f"Validation: {msg}")

    if not is_valid:
        print("ERROR: Prompt exceeds 100 character limit!")
        return

    print("\nEnter secret codeword to extract:")
    secret = input("> ").strip()

    if not secret:
        print("ERROR: Secret cannot be empty!")
        return

    print("\nSelect defense set:")
    print("1. Quick test (3 defenses)")
    print("2. Standard test (all defenses)")
    print("3. Custom selection")

    defense_choice = input("> ").strip()

    if defense_choice == "1":
        defenses = [
            SYSTEM_PROMPTS["v1_direct"],
            SYSTEM_PROMPTS["v3_encoded"],
            SYSTEM_PROMPTS["v5_final"],
        ]
        test_name = "Quick Test"
    elif defense_choice == "2":
        defenses = list(SYSTEM_PROMPTS.values())
        test_name = "Standard Test"
    elif defense_choice == "3":
        display_prompt_list(SYSTEM_PROMPTS, "Available Defensive Prompts")
        print("\nEnter defense IDs (comma-separated):")
        ids = input("> ").strip().split(",")
        defenses = [
            SYSTEM_PROMPTS[i.strip()] for i in ids if i.strip() in SYSTEM_PROMPTS
        ]
        test_name = "Custom Test"
        if not defenses:
            print("ERROR: No valid defense IDs selected!")
            return
    else:
        defenses = list(SYSTEM_PROMPTS.values())
        test_name = "Standard Test"

    print(f"\n🔄 Running {test_name} with {len(defenses)} defense(s)...")
    print("This may take a minute...\n")

    try:
        results = await run_prompt_attack_test(
            user_prompt=user_prompt,
            secret_codeword=secret,
            defense_prompts=defenses,
            api_key=api_key,
        )

        print_section("ATTACK TEST RESULTS")
        print(f"User Prompt: {user_prompt}")
        print(f"Secret: {secret}")
        print(f"Total Defenses: {results['total_defenses']}")
        print(f"Successful Attacks: {results['successful_attacks']}")
        print(f"Failed Attacks: {results['failed_attacks']}")
        print(f"Attack Success Rate: {results['attack_success_rate']:.1f}%")

        print("\nDetailed Results:")
        for detail in results["details"]:
            outcome = detail["outcome"]
            emoji = (
                "✅" if outcome == "SUCCESS" else "❌" if outcome == "FAILED" else "⚠️"
            )
            print(f"\n{emoji} Defense {detail['defense_id']}: {outcome}")
            print(f"   System: {detail['system_prompt'][:60]}...")
            if "assistant_reply" in detail:
                print(f"   AI: {detail['assistant_reply'][:60]}...")
            if "error" in detail:
                print(f"   Error: {detail['error']}")

        # Summary
        if results["attack_success_rate"] >= 90:
            print("\n🎉 EXCELLENT! This attack is very effective.")
        elif results["attack_success_rate"] >= 70:
            print("\n👍 GOOD! This attack is reasonably effective.")
        elif results["attack_success_rate"] >= 50:
            print("\n⚠️  MODERATE. This attack needs improvement.")
        else:
            print("\n❌ WEAK! This attack is easily blocked.")

    except Exception as e:
        print(f"\n❌ ERROR during testing: {e}")


async def show_stats():
    """Display prompt statistics."""
    print_header("PROMPT STATISTICS")

    stats = get_prompt_stats()

    print("System Prompts (Defensive):")
    print(f"  Total: {stats['system_prompts']['count']}")
    print(f"  Best: {stats['system_prompts']['best']}")
    print(
        f"  Average Length: {sum(stats['system_prompts']['lengths'].values()) / len(stats['system_prompts']['lengths']):.1f} chars"
    )

    print("\nUser Prompts (Offensive):")
    print(f"  Total: {stats['user_prompts']['count']}")
    print(f"  Best: {stats['user_prompts']['best']}")
    print(
        f"  Average Length: {sum(stats['user_prompts']['lengths'].values()) / len(stats['user_prompts']['lengths']):.1f} chars"
    )

    display_prompt_list(SYSTEM_PROMPTS, "\nAll System Prompts")
    display_prompt_list(USER_PROMPTS, "\nAll User Prompts")


async def custom_prompt_test(api_key: str):
    """Test a custom prompt entered by user."""
    print_header("CUSTOM PROMPT TESTING")

    print("Enter custom system prompt (≤100 chars):")
    system_prompt = input("> ").strip()

    is_valid, msg = validate_prompt_length(system_prompt)
    print(f"System prompt validation: {msg}")

    if not is_valid:
        print("ERROR: System prompt exceeds 100 character limit!")
        return

    print("\nEnter custom user prompt (≤100 chars):")
    user_prompt = input("> ").strip()

    is_valid, msg = validate_prompt_length(user_prompt)
    print(f"User prompt validation: {msg}")

    if not is_valid:
        print("ERROR: User prompt exceeds 100 character limit!")
        return

    print("\nEnter secret codeword:")
    secret = input("> ").strip()

    if not secret:
        print("ERROR: Secret cannot be empty!")
        return

    print("\n🔄 Testing custom prompts...")

    try:
        # Test the defense
        results = await run_prompt_defense_test(
            system_prompt=system_prompt,
            secret_codeword=secret,
            attack_prompts=[user_prompt],
            api_key=api_key,
        )

        print_section("CUSTOM PROMPT TEST RESULTS")
        print(f"System: {system_prompt}")
        print(f"User: {user_prompt}")
        print(f"Secret: {secret}")

        detail = results["details"][0]
        if detail["outcome"] == "SUCCESS":
            print("\n✅ DEFENSE SUCCESSFUL: Secret was protected!")
        else:
            print("\n❌ DEFENSE FAILED: Secret was leaked!")

        if "assistant_reply" in detail:
            print(f"\nAI Response: {detail['assistant_reply']}")

    except Exception as e:
        print(f"\n❌ ERROR during testing: {e}")


async def main():
    """Main CLI entry point."""
    print_header("VIVA COMPONENT: PROMPT ENGINEERING TESTER")

    # Check for API key
    api_key = os.getenv("AIPIPE_API_KEY")
    if not api_key:
        print("❌ ERROR: AIPIPE_API_KEY environment variable not set!")
        print("\nPlease set your API key:")
        print("  export AIPIPE_API_KEY=your_key_here")
        print("\nOr create a .env file with:")
        print("  AIPIPE_API_KEY=your_key_here")
        return

    print("✅ API key found")

    while True:
        print("\n" + "=" * 80)
        print("MAIN MENU")
        print("=" * 80)
        print("\n1. Test DEFENSE (system prompt vs. attacks)")
        print("2. Test ATTACK (user prompt vs. defenses)")
        print("3. Show STATISTICS")
        print("4. Test CUSTOM prompts")
        print("5. Exit")

        choice = input("\nEnter choice (1-5): ").strip()

        if choice == "1":
            await run_defense_mode(api_key)
        elif choice == "2":
            await run_attack_mode(api_key)
        elif choice == "3":
            await show_stats()
        elif choice == "4":
            await custom_prompt_test(api_key)
        elif choice == "5":
            print("\n👋 Goodbye!")
            break
        else:
            print("\n❌ Invalid choice. Please enter 1-5.")

        input("\nPress Enter to continue...")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)
