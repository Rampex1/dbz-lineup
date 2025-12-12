import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import textwrap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from pathlib import Path

# --- CONFIGURATION ---
BASE_DIR = Path(__file__).parent

INPUT_FILE = BASE_DIR / "factions.csv"
TEAM_HIGHLIGHT = "David"
OUTPUT_WEEKLY = BASE_DIR / "images/factions_weekly_progress.png"
OUTPUT_DISTRIBUTION = BASE_DIR / "images/factions_score_distribution.png"
OUTPUT_LEADERBOARD = BASE_DIR / "images/factions_top_10_mvp.png"


def load_data(filepath):
    """Loads and preprocesses the data."""
    try:
        df = pd.read_csv(filepath)
        return df
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        return None


def plot_team_progress(df):
    """
    Plots the weekly and cumulative points for each team on a single axis.
    """
    print("Generating Weekly Progress Graph...")

    # Define week columns
    week_cols = [f"Week {i}" for i in range(1, 9)]

    # Aggregate points
    team_weekly = df.groupby("Team")[week_cols].sum().T
    team_cumulative = team_weekly.cumsum()

    # Setup Plot
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(14, 8))
    palette = sns.color_palette("husl", len(team_weekly.columns))

    # Store handles for the legend
    lines = []
    labels = []

    # Plot lines
    for i, team in enumerate(team_weekly.columns):
        color = palette[i]

        # 1. Weekly Points (Solid)
        (l1,) = ax.plot(
            team_weekly.index,
            team_weekly[team],
            marker="o",
            linewidth=2.5,
            color=color,
            label=f"{team}",
        )

        # 2. Cumulative Points (Dashed & Lighter)
        ax.plot(
            team_cumulative.index,
            team_cumulative[team],
            marker="s",
            linestyle="--",
            linewidth=2,
            color=color,
            alpha=0.5,
        )

        lines.append(l1)
        labels.append(team)

    # Formatting
    ax.set_xlabel("Week", fontsize=14, labelpad=10)
    ax.set_ylabel("Points", fontsize=14, fontweight="bold")
    ax.set_title(
        "Team Points: Weekly & Cumulative", fontsize=18, fontweight="bold", pad=20
    )

    # Legend 1: Teams
    legend1 = ax.legend(lines, labels, title="Teams", loc="upper left", fontsize=12)

    # Legend 2: Line Styles (Manual)
    style_lines = [
        Line2D([0], [0], color="black", lw=2, label="Weekly Points"),
        Line2D(
            [0],
            [0],
            color="black",
            lw=2,
            linestyle="--",
            alpha=0.5,
            label="Cumulative Total",
        ),
    ]
    ax.legend(
        handles=style_lines, loc="upper left", bbox_to_anchor=(0, 0.75), fontsize=12
    )
    ax.add_artist(legend1)  # Add first legend back

    plt.tight_layout()
    plt.savefig(OUTPUT_WEEKLY)
    print(f"Saved: {OUTPUT_WEEKLY}")
    # plt.show() # Uncomment to display window


def plot_score_distribution(df):
    """
    Plots a stacked histogram of participant scores, highlighting a specific team.
    """
    print("Generating Score Distribution Graph...")

    # 1. Create Bins (0-9, 10-19, etc.)
    bins = range(0, 91, 10)
    labels = [f"{i}-{i + 9}" for i in bins[:-1]]
    # Use .copy() to avoid SettingWithCopyWarning if df is a slice
    df = df.copy()
    df["Score Range"] = pd.cut(df["TOTAL"], bins=bins, labels=labels, right=False)

    # 2. Pivot Data for Stacking
    pivot_counts = df.pivot_table(
        index="Score Range",
        columns="Team",
        values="Name",
        aggfunc="count",
        fill_value=0,
    )

    # Ensure Highlighted Team is Last (Top of Stack)
    teams = list(pivot_counts.columns)
    if TEAM_HIGHLIGHT in teams:
        teams.remove(TEAM_HIGHLIGHT)
        teams.append(TEAM_HIGHLIGHT)
    pivot_counts = pivot_counts[teams]

    # Get names for annotations
    highlight_names = (
        df[df["Team"] == TEAM_HIGHLIGHT].groupby("Score Range")["Name"].apply(list)
    )

    # Setup Plot
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(14, 10))

    bottom = np.zeros(len(pivot_counts))
    x_indices = np.arange(len(pivot_counts))

    # Consistent Colors
    all_teams_sorted = sorted(df["Team"].unique())
    palette = sns.color_palette("husl", len(all_teams_sorted))
    team_color_map = dict(zip(all_teams_sorted, palette))

    # 3. Create Stacked Bars
    for team in teams:
        counts = pivot_counts[team].values
        color = team_color_map.get(team, "gray")

        bar = ax.bar(
            x_indices,
            counts,
            width=0.8,
            bottom=bottom,
            label=team,
            color=color,
            edgecolor="white",
        )
        bottom += counts

        # 4. Annotations for Highlighted Team
        if team == TEAM_HIGHLIGHT:
            for i, rect in enumerate(bar):
                height = rect.get_height()
                if height > 0:
                    bin_label = pivot_counts.index[i]

                    # Logic: Hide 0-9 range
                    if bin_label == "0-9":
                        continue

                    # Logic: Get names and filter 'Challenge'
                    names = highlight_names.get(bin_label, [])
                    names = [n for n in names if "challenge" not in n.lower()]

                    if not names:
                        continue

                    # Create Label
                    wrapped_names = textwrap.fill(", ".join(names), width=20)

                    # Position
                    ax.annotate(
                        wrapped_names,
                        xy=(rect.get_x() + rect.get_width() / 2, rect.get_y() + height),
                        xytext=(0, 5),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=9,
                        fontweight="bold",
                        bbox=dict(
                            boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9
                        ),
                    )

    # Formatting
    ax.set_xticks(x_indices)
    ax.set_xticklabels(pivot_counts.index, fontsize=12)
    ax.set_xlabel("Total Point Range", fontsize=14, fontweight="bold", labelpad=10)
    ax.set_ylabel("Number of Participants", fontsize=14, fontweight="bold", labelpad=10)
    ax.set_title(
        "Participant Distribution by Score Range",
        fontsize=18,
        fontweight="bold",
        pad=20,
    )

    # Legend (Reversed to match stack order)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles[::-1], labels[::-1], title="Teams", fontsize=12, loc="upper right"
    )

    plt.tight_layout()
    plt.savefig(OUTPUT_DISTRIBUTION)
    print(f"Saved: {OUTPUT_DISTRIBUTION}")
    # plt.show() # Uncomment to display window


def plot_top_10_leaderboard(df):
    """
    Plots a horizontal bar chart of the top 10 highest scorers across all teams.
    """
    print("Generating Top 10 Leaderboard...")

    # 1. Filter out 'Challenge' entries (case-insensitive)
    df_filtered = df[~df["Name"].str.contains("Challenge", case=False, na=False)].copy()

    # 2. Sort by TOTAL points (Descending) and take Top 10
    top_10 = df_filtered.sort_values("TOTAL", ascending=False).head(10)

    # Reverse for plotting (so #1 is at the top)
    top_10 = top_10.iloc[::-1]

    # 3. Setup Plot
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, 8))

    # Color Palette - Consistent
    all_teams_sorted = sorted(df["Team"].unique())
    palette = sns.color_palette("husl", len(all_teams_sorted))
    team_color_map = dict(zip(all_teams_sorted, palette))

    # Map colors
    colors = top_10["Team"].map(team_color_map)

    # 4. Create Horizontal Bar Chart
    bars = ax.barh(top_10["Name"], top_10["TOTAL"], color=colors)

    # 5. Add Value Labels
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{int(width)}",
            va="center",
            fontsize=12,
            fontweight="bold",
        )

    # 6. Formatting
    ax.set_xlabel("Total Points", fontsize=14, fontweight="bold")
    ax.set_ylabel("Participant", fontsize=14, fontweight="bold")
    ax.set_title("Top 10 MVP Leaderboard", fontsize=18, fontweight="bold", pad=20)

    # Custom Legend for Teams
    legend_elements = [
        Patch(facecolor=team_color_map[team], label=team) for team in all_teams_sorted
    ]
    ax.legend(handles=legend_elements, title="Teams", loc="lower right", fontsize=12)

    ax.set_xlim(0, top_10["TOTAL"].max() * 1.1)

    plt.tight_layout()
    plt.savefig(OUTPUT_LEADERBOARD)
    print(f"Saved: {OUTPUT_LEADERBOARD}")
    # plt.show() # Uncomment to display window

    # 7. Print complete ranking
    print("\n" + "=" * 50)
    print("COMPLETE RANKING - ALL PARTICIPANTS")
    print("=" * 50)
    full_ranking = df_filtered.sort_values("TOTAL", ascending=False)
    for rank, (idx, row) in enumerate(full_ranking.iterrows(), start=1):
        print(f"{rank:3d}. {row['Name']:<30s} {int(row['TOTAL']):>4d} points")
    print("=" * 50 + "\n")


def create_balanced_teams(df, num_teams=5):
    """
    Creates balanced teams based on total points.
    Constraint: Meng Jia, David Zhou, Feng, Orlando, Vincent must be on different teams.
    """
    print("Creating Balanced Teams...")

    # 1. Filter out 'Challenge' entries
    df_filtered = df[~df["Name"].str.contains("Challenge", case=False, na=False)].copy()

    # 2. Sort by TOTAL points (Descending)
    df_sorted = df_filtered.sort_values("TOTAL", ascending=False).reset_index(drop=True)

    # 3. Define constrained players (must be on different teams)
    constrained_players = ["Meng Jia", "David Zhou", "Feng", "Orlando", "Vincent"]

    # 4. Initialize teams
    teams = {f"Team {i + 1}": {"players": [], "total_points": 0} for i in range(num_teams)}
    team_names = list(teams.keys())

    # 5. Assign constrained players to different teams first
    constrained_assigned = []
    for i, player_name in enumerate(constrained_players):
        # Find player in dataframe (case-insensitive partial match)
        player_row = df_sorted[df_sorted["Name"].str.contains(player_name, case=False, na=False)]
        if not player_row.empty:
            player_row = player_row.iloc[0]
            team_name = team_names[i % num_teams]
            teams[team_name]["players"].append({
                "name": player_row["Name"],
                "points": player_row["TOTAL"]
            })
            teams[team_name]["total_points"] += player_row["TOTAL"]
            constrained_assigned.append(player_row["Name"])
            print(f"Assigned {player_row['Name']} ({int(player_row['TOTAL'])} pts) to {team_name}")

    # 6. Get remaining players
    remaining_players = df_sorted[~df_sorted["Name"].isin(constrained_assigned)].copy()

    # 7. Assign remaining players using a balanced approach (snake draft)
    for idx, row in remaining_players.iterrows():
        # Find team with lowest total points
        min_team = min(teams.items(), key=lambda x: x[1]["total_points"])
        team_name = min_team[0]

        teams[team_name]["players"].append({
            "name": row["Name"],
            "points": row["TOTAL"]
        })
        teams[team_name]["total_points"] += row["TOTAL"]

    # 8. Print results
    print("\n" + "=" * 70)
    print("BALANCED TEAM ASSIGNMENTS")
    print("=" * 70)

    for team_name in team_names:
        team = teams[team_name]
        print(f"\n{team_name} - Total Points: {int(team['total_points'])}")
        print("-" * 70)
        for i, player in enumerate(team["players"], start=1):
            print(f"  {i:2d}. {player['name']:<40s} {int(player['points']):>4d} pts")

    print("\n" + "=" * 70)
    print("TEAM POINT SUMMARY")
    print("=" * 70)
    for team_name in team_names:
        print(f"{team_name}: {int(teams[team_name]['total_points']):>5d} points "
              f"({len(teams[team_name]['players'])} players)")

    # Calculate balance metrics
    total_points = [teams[team]["total_points"] for team in team_names]
    avg_points = sum(total_points) / len(total_points)
    max_diff = max(total_points) - min(total_points)
    print(f"\nAverage Team Points: {int(avg_points)}")
    print(f"Max Point Difference: {int(max_diff)}")
    print("=" * 70 + "\n")

    return teams

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    df = load_data(INPUT_FILE)
    if df is not None:
        plot_team_progress(df)
        plot_score_distribution(df)
        plot_top_10_leaderboard(df)
        create_balanced_teams(df)

        # Display the plots at the end if you want them to pop up one by one
        plt.show()
