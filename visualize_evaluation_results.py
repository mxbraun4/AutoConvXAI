#!/usr/bin/env python3
"""Generate visualizations for evaluation results."""

import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Load results
with open('evaluation_results_final.json', 'r') as f:
    data = json.load(f)
    report = data['report']

# Create figure with subplots
fig = plt.figure(figsize=(16, 12))
fig.suptitle('TalkToModel Evaluation Results - Comprehensive Analysis', fontsize=20, fontweight='bold')

# 1. Overall Accuracy Metrics (Top Left)
ax1 = plt.subplot(2, 3, 1)
metrics = ['Action\nAccuracy', 'Entity\nAccuracy', 'Overall\nAccuracy']
values = [report['action_accuracy'], report['entity_accuracy'], report['overall_accuracy']]
percentages = [v * 100 for v in values]

bars = ax1.bar(metrics, percentages, color=['#2E86AB', '#A23B72', '#F18F01'])
ax1.set_ylim(0, 105)
ax1.set_ylabel('Accuracy (%)', fontsize=12)
ax1.set_title('Overall System Performance', fontsize=14, fontweight='bold')

# Add value labels on bars
for bar, pct in zip(bars, percentages):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{pct:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Add grid
ax1.grid(axis='y', alpha=0.3)

# 2. Action-wise Performance (Top Right)
ax2 = plt.subplot(2, 3, 2)
action_data = report['action_breakdown']
actions = list(action_data.keys())
accuracies = [action_data[a]['accuracy'] * 100 for a in actions]
totals = [action_data[a]['total'] for a in actions]

# Sort by accuracy
sorted_indices = np.argsort(accuracies)
actions = [actions[i] for i in sorted_indices]
accuracies = [accuracies[i] for i in sorted_indices]
totals = [totals[i] for i in sorted_indices]

# Create horizontal bar chart
y_pos = np.arange(len(actions))
bars = ax2.barh(y_pos, accuracies)

# Color bars based on accuracy
colors = ['#d32f2f' if acc < 85 else '#f57c00' if acc < 95 else '#388e3c' for acc in accuracies]
for bar, color in zip(bars, colors):
    bar.set_color(color)

ax2.set_yticks(y_pos)
ax2.set_yticklabels(actions)
ax2.set_xlabel('Accuracy (%)', fontsize=12)
ax2.set_xlim(0, 105)
ax2.set_title('Performance by Action Type', fontsize=14, fontweight='bold')

# Add value labels
for i, (acc, total) in enumerate(zip(accuracies, totals)):
    ax2.text(acc + 1, i, f'{acc:.1f}% (n={total})', va='center', fontsize=10)

# 3. Test Case Distribution (Middle Left)
ax3 = plt.subplot(2, 3, 3)
action_counts = [action_data[a]['total'] for a in action_data.keys()]
action_names = list(action_data.keys())

# Create pie chart
colors_pie = plt.cm.Set3(range(len(action_names)))
wedges, texts, autotexts = ax3.pie(action_counts, labels=action_names, autopct='%1.0f%%',
                                    colors=colors_pie, startangle=90)
ax3.set_title('Test Case Distribution by Action', fontsize=14, fontweight='bold')

# Make percentage text smaller
for autotext in autotexts:
    autotext.set_fontsize(9)
    autotext.set_fontweight('bold')

# 4. Accuracy Comparison Matrix (Middle Right)
ax4 = plt.subplot(2, 3, 4)
categories = ['Perfect\n(100%)', 'High\n(90-99%)', 'Good\n(80-89%)', 'Need Work\n(<80%)']
action_categories = [0, 0, 0, 0]

for action, data in action_data.items():
    acc = data['accuracy'] * 100
    if acc == 100:
        action_categories[0] += 1
    elif acc >= 90:
        action_categories[1] += 1
    elif acc >= 80:
        action_categories[2] += 1
    else:
        action_categories[3] += 1

bars = ax4.bar(categories, action_categories, color=['#4caf50', '#8bc34a', '#ff9800', '#f44336'])
ax4.set_ylabel('Number of Action Types', fontsize=12)
ax4.set_title('Action Performance Distribution', fontsize=14, fontweight='bold')
ax4.set_ylim(0, max(action_categories) + 2)

# Add value labels
for bar in bars:
    height = bar.get_height()
    if height > 0:
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{int(height)}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# 5. Validation Impact (Bottom Left)
ax5 = plt.subplot(2, 3, 5)
validation_impact = report.get('validation_impact', {})
if validation_impact.get('cases_improved', 0) > 0 or validation_impact.get('cases_worsened', 0) > 0:
    impact_labels = ['Improved', 'No Change', 'Worsened']
    impact_values = [
        validation_impact.get('cases_improved', 0),
        validation_impact.get('no_change', 0),
        validation_impact.get('cases_worsened', 0)
    ]
    colors_impact = ['#4caf50', '#2196f3', '#f44336']
    
    bars = ax5.bar(impact_labels, impact_values, color=colors_impact)
    ax5.set_ylabel('Number of Cases', fontsize=12)
    ax5.set_title('Validation Agent Impact', fontsize=14, fontweight='bold')
    
    # Add percentage labels
    total_cases = sum(impact_values)
    for bar, val in zip(bars, impact_values):
        if val > 0:
            pct = (val / total_cases) * 100
            ax5.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                     f'{val}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=10)
else:
    ax5.text(0.5, 0.5, 'No Validation Changes Detected', 
             ha='center', va='center', fontsize=14, transform=ax5.transAxes)
    ax5.set_xticks([])
    ax5.set_yticks([])

# 6. Key Statistics Summary (Bottom Right)
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')

# Create summary text
summary_text = f"""Key Evaluation Statistics

Total Test Cases: {report['total_cases']}
Error Rate: {report['error_rate'] * 100:.1f}%

Top Performing Actions:
"""

# Find top 3 performing actions
sorted_actions = sorted(action_data.items(), key=lambda x: x[1]['accuracy'], reverse=True)[:3]
for i, (action, data) in enumerate(sorted_actions, 1):
    summary_text += f"\n{i}. {action.capitalize()}: {data['accuracy']*100:.1f}% ({data['total']} cases)"

summary_text += "\n\nAreas for Improvement:\n"
# Find bottom 3 performing actions
bottom_actions = sorted(action_data.items(), key=lambda x: x[1]['accuracy'])[:3]
for i, (action, data) in enumerate(bottom_actions, 1):
    if data['accuracy'] < 1.0:  # Only show if not perfect
        summary_text += f"\n{i}. {action.capitalize()}: {data['accuracy']*100:.1f}% ({data['total']} cases)"

ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, fontsize=12,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))

plt.tight_layout()
plt.savefig('evaluation_results_visualization.png', dpi=300, bbox_inches='tight')
plt.savefig('evaluation_results_visualization.pdf', bbox_inches='tight')
print("Visualizations saved as 'evaluation_results_visualization.png' and '.pdf'")

# Create a second figure for detailed action analysis
fig2, ax = plt.subplots(figsize=(12, 8))

# Create heatmap-style visualization
actions_list = list(action_data.keys())
metrics = ['Total Cases', 'Correct', 'Accuracy (%)']
data_matrix = []

for action in actions_list:
    data_matrix.append([
        action_data[action]['total'],
        action_data[action]['correct'],
        action_data[action]['accuracy'] * 100
    ])

data_matrix = np.array(data_matrix).T

# Create heatmap
im = ax.imshow(data_matrix, cmap='YlOrRd', aspect='auto')

# Set ticks and labels
ax.set_xticks(np.arange(len(actions_list)))
ax.set_yticks(np.arange(len(metrics)))
ax.set_xticklabels(actions_list, rotation=45, ha='right')
ax.set_yticklabels(metrics)

# Add text annotations
for i in range(len(metrics)):
    for j in range(len(actions_list)):
        value = data_matrix[i, j]
        if i == 2:  # Accuracy
            text = ax.text(j, i, f'{value:.1f}%', ha='center', va='center', color='black')
        else:
            text = ax.text(j, i, f'{int(value)}', ha='center', va='center', color='black')

ax.set_title('Detailed Action Performance Matrix', fontsize=16, fontweight='bold', pad=20)
fig2.colorbar(im, ax=ax, label='Value Scale')

plt.tight_layout()
plt.savefig('evaluation_action_matrix.png', dpi=300, bbox_inches='tight')
print("Action matrix saved as 'evaluation_action_matrix.png'")

plt.show()