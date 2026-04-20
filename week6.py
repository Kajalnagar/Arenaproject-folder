import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

df = sns.load_dataset("tips")

print("First 5 rows of dataset:")
print(df.head())

sns.set_style("darkgrid")
sns.set_palette("Set2")

plt.figure(figsize=(6,4))
sns.scatterplot(x="total_bill", y="tip", hue="day", data=df)
plt.title("Scatter Plot: Total Bill vs Tip")
plt.show()

plt.figure(figsize=(6,4))
sns.lineplot(x="size", y="total_bill", data=df)
plt.title("Line Plot: Party Size vs Total Bill")
plt.show()

plt.figure(figsize=(6,4))
sns.boxplot(x="day", y="total_bill", data=df)
plt.title("Box Plot: Total Bill Distribution by Day")
plt.show()

plt.figure(figsize=(6,4))
sns.violinplot(x="day", y="tip", data=df)
plt.title("Violin Plot: Tip Distribution by Day")
plt.show()

plt.figure(figsize=(6,4))
sns.barplot(x="day", y="tip", data=df)
plt.title("Bar Plot: Average Tip per Day")
plt.show()

plt.figure(figsize=(6,4))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

fig, axes = plt.subplots(2,2, figsize=(12,8))

sns.scatterplot(x="total_bill", y="tip", data=df, ax=axes[0,0])
axes[0,0].set_title("Scatter Plot")

sns.boxplot(x="day", y="total_bill", data=df, ax=axes[0,1])
axes[0,1].set_title("Box Plot")

sns.violinplot(x="day", y="tip", data=df, ax=axes[1,0])
axes[1,0].set_title("Violin Plot")

sns.barplot(x="day", y="tip", data=df, ax=axes[1,1])
axes[1,1].set_title("Bar Plot")

plt.tight_layout()
plt.show()

fig = px.scatter(
    df,
    x="total_bill",
    y="tip",
    color="day",
    size="size",
    hover_data=["sex", "smoker"],
    title="Interactive Scatter Plot"
)

fig.show()

fig = px.bar(
    df,
    x="day",
    y="tip",
    color="day",
    title="Interactive Bar Chart: Tips by Day"
)

fig.show()

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=("Scatter", "Box", "Histogram", "Bar")
)

fig.add_trace(
    go.Scatter(
        x=df["total_bill"],
        y=df["tip"],
        mode='markers',
        name="Scatter"
    ),
    row=1, col=1
)

fig.add_trace(
    go.Box(
        y=df["total_bill"],
        name="Total Bill"
    ),
    row=1, col=2
)

fig.add_trace(
    go.Histogram(
        x=df["tip"],
        name="Tip Distribution"
    ),
    row=2, col=1
)

fig.add_trace(
    go.Bar(
        x=df["day"],
        y=df["tip"],
        name="Tips"
    ),
    row=2, col=2
)

fig.update_layout(
    title="Interactive Data Visualization Dashboard",
    height=700
)

fig.show()

plt.savefig("dashboard.png")

print("Dashboard project completed successfully!")
