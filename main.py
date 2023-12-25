import PySimpleGUI as sg
from neural_network import NeuralNetwork
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

font_style = ('Helvetica', 16, 'bold')
nn = None

def create_layout():
    activation_functions = ['Sigmoid', 'Tanh', 'ReLU', 'Leaky ReLU', 'ELU', 'Linear']
    output_activation_functions = ['Sigmoid', 'Tanh', 'ReLU', 'Leaky ReLU', 'ELU', 'Softmax', 'Linear']
    left_column = [
        [sg.Text('Data Set:', font=font_style)],
        [sg.InputText(key='data_set', size=(25, 1), disabled=True, text_color='#fff', font=font_style),
         sg.FileBrowse(file_types=(("CSV Files", "*.csv"),), key='file_browse')],
        [sg.Text('Number of neurons in hidden layer:', font=font_style)],
        [sg.InputText(default_text='8', key='neurons', font=font_style)],
        [sg.Text('Learning rate:', font=font_style)],
        [sg.InputText(default_text='0.01', key='learning_rate', font=font_style)],
        [sg.Text('Maximum number of epochs:', font=font_style)],
        [sg.InputText(default_text='100', key='max_epochs', font=font_style)],
        [sg.Text('Hidden Activation Function:', font=font_style)],
        [sg.DropDown(activation_functions, default_value='Sigmoid', key='hidden_activation_function', font=font_style)],
        [sg.Text('Output Activation Function:', font=font_style)],
        [sg.DropDown(output_activation_functions, default_value='Sigmoid', key='output_activation_function', font=font_style)],
        [sg.Text('Goal:', font=font_style)],
        [sg.Slider(range=(0, 100), orientation='h', size=(20, 20), default_value=50, key='goal', font=font_style)],
        [sg.Text('', size=(30, 1))],
        [sg.Button('Train', font=font_style), sg.Button('Exit', font=font_style)],
    ]

    right_column = [
        [sg.Text('Progress:', font=font_style)],
        [sg.ProgressBar(100, orientation='h', size=(40, 20), key='progress_bar')],
        [sg.Text('', background_color='blue', text_color='yellow', font=('Helvetica', 16), key='model_accuracy')],
        [sg.Button('Predict', key='PredictBtn', font=font_style, visible=False), sg.Button('Test with file', key='TestWithFileBtn', font=font_style, visible=False)],
    ]

    layout = [
        [
            sg.Column(left_column, element_justification='left', size=(400, 500)),
            sg.Column(right_column, element_justification='center', size=(400, 500))
        ]
    ]

    return layout

def start_training(file_browse, neurons, learning_rate, max_epochs, hidden_activation_function,
                   output_activation_function, goal, window):
    global nn
    dataset = pd.read_csv(file_browse)
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values
    # Convert string labels to integers using LabelEncoder
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    def on_progress_cal(output):
        window['progress_bar'].update_bar((output/int(max_epochs)) * 100)
    
    # Initialize and train the neural network
    nn = NeuralNetwork(
        hidden_size=int(neurons),
        activation_function=hidden_activation_function,
        output_function=output_activation_function,
        epochs=int(max_epochs),
        learning_rate=float(learning_rate),
        goal=float(goal)/100,
        progress_cal=on_progress_cal
    )
    nn.train(X_train, y_train)

    # Predict on the test set
    predictions = nn.predict(X_test)

    # Calculate and print accuracy
    accuracy = accuracy_score(y_test, predictions)
    window['model_accuracy'].update(f'Test Accuracy: {accuracy:.2%}')

def show_input_columns(file_path):
    global nn
    # Read the dataset
    dataset = pd.read_csv(file_path)
    label_encoder = LabelEncoder()
    label_encoder.fit(dataset.iloc[:, -1].values)
    # Create a new window for the form
    form_layout = [
        [sg.Text('Predict:')],
    ]

    input_fields = []

    for column in dataset.columns[:-1]:
        form_layout.append([sg.Text(f'{column}:', size=(15, 1)), sg.InputText(key=column)])
        input_fields.append(column)

    form_layout.append([sg.Button('Submit'), sg.Button('Cancel')])
    form_layout.append([sg.Text('', size=(30, 1), key='prediction_text')])

    form_window = sg.Window('Input Form', form_layout, modal=True)

    while True:
        event, values = form_window.read()

        if event == sg.WINDOW_CLOSED or event == 'Cancel':
            break

        if event == 'Submit':
            # Get values from input fields
            input_values = [float(values[column]) for column in input_fields]
            # Perform prediction using the neural network
            prediction_class = nn.predict([input_values])
            predicted_label = label_encoder.inverse_transform([prediction_class])[0]
            form_window['prediction_text'].update(f'Predicted Label: {predicted_label}')

    form_window.close()

def test_with_file(test_file):
    global nn
    if(nn is None):
        return
    dataset = pd.read_csv(test_file)
    X = dataset.iloc[:, :-1].values
    print(X)
    y = dataset.iloc[:, -1].values
    # Convert string labels to integers using LabelEncoder
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    predictions = nn.predict(X)
    accuracy = accuracy_score(y_encoded, predictions)

    # Populate the table with results
    table_data = []
    for i, (input_values, actual_class, predicted_class) in enumerate(zip(X, y, predictions), start=1):
        row_data = [i] + input_values.tolist() + [actual_class, label_encoder.inverse_transform([predicted_class])[0]]
        table_data.append(row_data)

    # Create a new window for the results table
    table_layout = [
        [sg.Text(f'Model Accuracy on Test File: {accuracy:.2%}')],
        [sg.Table(values=table_data,
                  headings=['Index'] + dataset.columns.tolist()[:-1] + ['Actual Class', 'Predicted Class'],
                  auto_size_columns=False,
                  justification='right',
                  display_row_numbers=False,
                  num_rows=min(25, len(X)),
                  key='-TABLE-')],
        [sg.Button('Exit', font=font_style)]
    ]

    table_window = sg.Window('Results Table', table_layout)

    while True:
        event, _ = table_window.read()

        if event == sg.WINDOW_CLOSED or event == 'Exit':
            break

    table_window.close()
    
def main():
    sg.theme('BluePurple')
    window = sg.Window('Multi-class Classification By Wajed Afaneh', create_layout())

    while True:
        event, values = window.read()

        if event == sg.WINDOW_CLOSED or event == 'Exit':
            break

        if event == 'Train':
            window['progress_bar'].update_bar(0)
            start_training(
                values['file_browse'],
                values['neurons'],
                values['learning_rate'],
                values['max_epochs'],
                values['hidden_activation_function'],
                values['output_activation_function'],
                values['goal'],
                window,
            )
            window['PredictBtn'].update(visible=True)
            window['TestWithFileBtn'].update(visible=True)

        if event == 'file_browse':
            selected_file_path = sg.popup_get_file('Select a CSV file', file_types=(("CSV Files", "*.csv"),))
            window['data_set'].update(value=selected_file_path)
            
        if event == 'TestWithFileBtn':
            test_file = sg.popup_get_file('Select a CSV file for testing', file_types=(("CSV Files", "*.csv"),))
            test_with_file(test_file)
            
        if event == 'PredictBtn':
            show_input_columns(values['file_browse'])
            
    window.close()


if __name__ == '__main__':
    main()
