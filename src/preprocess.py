
class Preprocessing:
    def execute(self, data):

        # Dropping unuseful columns
        cols = ['Ticket', "Cabin", "PassengerId"]
        data = data.drop(cols, axis=1)

        # Dropping rows with missing values
        data = data.dropna()

        return data
