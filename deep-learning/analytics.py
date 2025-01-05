import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from datetime import datetime
import os

class PerformanceAnalytics:
    def __init__(self):
        # Create data directory if it doesn't exist
        self.results_dir = 'analysis_results'
        os.makedirs(self.results_dir, exist_ok=True)
        
    def computational_comparison(self):
        # Data for comparison
        methods = ['XFEM', 'CNN']
        processing_times = [3600, 5]  # in seconds
        memory_usage = [24, 3]  # in GB (using average values)
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Processing Time Comparison (Log scale)
        ax1.bar(methods, processing_times, color=['#FF9999', '#66B2FF'])
        ax1.set_ylabel('Processing Time (seconds)')
        ax1.set_title('Processing Time Comparison (Log Scale)')
        ax1.set_yscale('log')  # Use log scale due to large difference
        
        # Add value labels on bars
        for i, v in enumerate(processing_times):
            ax1.text(i, v, f'{v}s', ha='center', va='bottom')
        
        # Plot 2: Memory Usage Comparison
        ax2.bar(methods, memory_usage, color=['#FF9999', '#66B2FF'])
        ax2.set_ylabel('Memory Usage (GB)')
        ax2.set_title('Memory Usage Comparison')
        
        # Add value labels on bars
        for i, v in enumerate(memory_usage):
            ax2.text(i, v, f'{v}GB', ha='center', va='bottom')
        
        # Overall plot settings
        plt.suptitle('Computational Efficiency: XFEM vs CNN', fontsize=16)
        plt.tight_layout()
        
        # Save the plot
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(f'{self.results_dir}/computational_comparison_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()

    def detailed_performance_analysis(self):
        # Create more detailed performance data
        data = {
            'Method': ['XFEM', 'XFEM', 'XFEM', 'CNN', 'CNN', 'CNN'],
            'Sample_Size': ['Small', 'Medium', 'Large', 'Small', 'Medium', 'Large'],
            'Processing_Time': [1800, 3600, 7200, 2, 5, 10],
            'Memory_Usage': [16, 24, 32, 2, 3, 4]
        }
        
        df = pd.DataFrame(data)
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Processing Time by Sample Size
        sns.barplot(data=df, x='Sample_Size', y='Processing_Time', hue='Method', ax=ax1)
        ax1.set_ylabel('Processing Time (seconds)')
        ax1.set_title('Processing Time by Sample Size')
        ax1.set_yscale('log')
        
        # Plot 2: Memory Usage by Sample Size
        sns.barplot(data=df, x='Sample_Size', y='Memory_Usage', hue='Method', ax=ax2)
        ax2.set_ylabel('Memory Usage (GB)')
        ax2.set_title('Memory Usage by Sample Size')
        
        # Overall plot settings
        plt.suptitle('Detailed Performance Analysis: XFEM vs CNN', fontsize=16)
        plt.tight_layout()
        
        # Save the plot
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(f'{self.results_dir}/detailed_performance_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()

    def efficiency_ratio_analysis(self):
        # Calculate efficiency ratios
        sample_sizes = ['Small', 'Medium', 'Large']
        xfem_times = [1800, 3600, 7200]
        cnn_times = [2, 5, 10]
        
        efficiency_ratios = [x/y for x, y in zip(xfem_times, cnn_times)]
        
        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.plot(sample_sizes, efficiency_ratios, marker='o', linewidth=2, markersize=10)
        plt.fill_between(sample_sizes, efficiency_ratios, alpha=0.2)
        
        plt.ylabel('Efficiency Ratio (XFEM Time / CNN Time)')
        plt.title('Efficiency Ratio Analysis')
        plt.grid(True, alpha=0.3)
        
        # Add value labels
        for i, ratio in enumerate(efficiency_ratios):
            plt.text(i, ratio, f'{ratio:.0f}x', ha='center', va='bottom')
        
        # Save the plot
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(f'{self.results_dir}/efficiency_ratio_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    analytics = PerformanceAnalytics()
    
    # Generate all plots
    analytics.computational_comparison()
    analytics.detailed_performance_analysis()
    analytics.efficiency_ratio_analysis()
    
    print("Analytics completed. Plots saved in 'analysis_results' directory.")

if __name__ == "__main__":
    main()
