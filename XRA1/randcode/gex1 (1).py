import matplotlib.pyplot as plt
import pandas as pd

# Put all the logic in this class. Do not change the class name, as the test cases depend on it.


class DataInspection:
    # Initialize the DataFrame. Do not change this method.
    def __init__(self):
        self.df = None
        self.numeric_cols = []
        self.ordinal_cols = []

    def load_csv(self, file_path):
        """
        Put here the logic to a CSV file into a DataFrame.
        """
        self.df = pd.read_csv(file_path)

    def handle_missing_values(self, col):
        """
        Handle missing values by imputing or dropping columns with too many missing values.
        If more than 50% of values are missing in a column, drop the column.
        Otherwise, impute based on column type.
        """

        missing_percentage = self.df[col].isna().mean() * 100
        no_column_dropped = True

        if missing_percentage > 50:
            print(f"Dropping column '{col}' (because more than 50% missing)")
            self.df.drop(columns=col, inplace=True)
            no_column_dropped = False
        
        elif missing_percentage > 0:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                replacement_value = self.df[col].median()
                
            else:
                replacement_value = self.df[col].mode()[0]

            self.df[col] = self.df[col].fillna(replacement_value)
        
        return no_column_dropped



    def check_data_types(self, col):
        """
        Check for incorrect data types and attempt to fix them.
        For example, convert numeric-looking strings to actual numeric types.
        """

        if pd.api.types.is_object_dtype(self.df[col]):
            try:
                self.df[col] = pd.to_numeric(self.df[col])
            except ValueError as err:
                print(f" Error converting {col} datatype to numeric. Column is a string")

    def classify_and_calculate(self, col):
        """
        Classifies the column's data type, calculates the central tendency like mean, median, and mode.
        """

        self.handle_missing_values(col)
        self.check_data_types(col)
        unique_values = self.df[col].nunique()

        if pd.api.types.is_numeric_dtype(self.df[col]):
            self.numeric_cols.append(col)

            if unique_values < 10:
                value = self.df[col].median()
            
            else:
                value = self.df[col].mean()
        
        elif pd.api.types.is_object_dtype(self.df[col]):
            self.ordinal_cols.append(col)
            value = self.df[col].mode()[0]
        
        return value
        


    # Loop through each column in the DataFrame and apply classification and plotting
    def classify_columns(self):
        """Loop through each column in the DataFrame and apply classification and central tendency calculation functions."""

        for col in self.df.columns:
            self.classify_and_calculate(col)
    
    def plot_histogram(self, col):
        """
        Function that plots a histogram for a given column
        """

        self.df[col].plot(kind="hist", title=f"Histogram of {col}")
        plt.xlabel(col)
        plt.show()

    def plot_box_plot(self, x_col, y_col):
        """
        Function that plots a box plot for given columns
        """

        self.df.boxplot(column=self.numeric_cols[y_col], by=self.ordinal_cols[x_col])
        plt.title(f'Box Plot of {self.numeric_cols[y_col]} by {self.ordinal_cols[x_col]}')
        plt.suptitle('')  # Removes the default Pandas title
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.show()

    def plot_bar_chart(self, col):
        """Funtiona that plots a bar plot for a given column"""

        self.df[col].value_counts().plot(kind='bar', title=f'Bar Chart of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.show()
    
    def plot_scatter(self, x_col, y_col):
        """Plots a scatter plot"""

        self.df.plot.scatter(x=x_col, y=y_col, title=f'Scatter Plot of {x_col} vs {y_col}')
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.show()

    def ask_for_histogram(self):
        """
        Interactive histogram function that prompts the user to select a column to plot a histogram.
        """

        for i, col in enumerate(self.numeric_cols):
            print(f"{i+1}. {col}")
        
        num = int(input("Enter column number: ")) - 1 

        self.plot_histogram(self.numeric_cols[num])



    def ask_for_boxplot(self):
        """
        Interactive box plot function that prompts the user to select columns for the X and Y axes.
        The function then plots a box plot of the selected columns.
        """

        print("\nOrdinal/Nominal columns available for X-axis:")
        # 'enumerate' is used to get the index of the column along with the column name
        for i, col in enumerate(self.ordinal_cols):
            print(f"{i+1}: {col}")

        print("\nNumeric columns available for Y-axis:")
        for i, col in enumerate(self.numeric_cols):
            print(f"{i+1}: {col}")

        x_idx = int(input("Enter the number for the X-axis column: ")) - 1
        y_idx = int(input("Enter the number for the Y-axis column: ")) - 1

        self.plot_box_plot(x_idx, y_idx)

    def ask_for_barplot(self):
        """
        Interactive bar plot function that prompts the user to select a column to plot a bar plot.
        """

        for i, col in enumerate(self.ordinal_cols):
            print(f"{i+1}: {col}")

        num_col = int(input("Enter the number for the X-axis column: ")) - 1

        self.plot_bar_chart(self.ordinal_cols[num_col])

    def ask_for_scatterplot(self):
        """
        Interactive scatterplot function that prompts the user to select columns for the X and Y axes. 
        The function then plots a scatterplot of the selected columns.
        """

        print("\nNumeric columns available for scatterplot:")
        for i, col in enumerate(self.numeric_cols):
            print(f"{i+1}: {col}")

        x_idx = int(input("Enter the number for the X-axis column: ")) - 1
        y_idx = int(input("Enter the number for the Y-axis column: ")) - 1

        self.plot_scatter(self.numeric_cols[x_idx], self.numeric_cols[y_idx])


# Main function
def main():
    """
    First create an object
    Then call the load_csv method
    Then call the classify_columns method
    Then call the ask_for_histogram method
    Then call the ask_for_boxplot method
    Then call the ask_for_barplot method
    Then call the ask_for_scatterplot method
    """

    analysis = DataInspection()
    path = input("Provide path to dataset: ")
    analysis.load_csv(path)
    analysis.classify_columns()
    analysis.ask_for_histogram()
    analysis.ask_for_boxplot()
    analysis.ask_for_barplot()
    analysis.ask_for_scatterplot()



# This is needed to run the main function when this script is run directly. Do not change this part.
if __name__ == "__main__":
    main()
