import PySimpleGUI as sg
from neural_network import NeuralNetwork
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

def create_layout():
    activation_functions = ['Sigmoid', 'Tanh', 'ReLU', 'Leaky ReLU', 'ELU', 'Linear']
    output_activation_functions = ['Sigmoid', 'Tanh', 'ReLU', 'Leaky ReLU', 'ELU', 'Softmax', 'Linear']
    font_style = ('Helvetica', 16, 'bold')
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
    dataset = pd.read_csv(file_browse)
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values
    # Convert string labels to integers using LabelEncoder
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    def on_progress_cal(output):
        window['progress_bar'].update_bar(output)
    
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

        if event == 'file_browse':
            selected_file_path = sg.popup_get_file('Select a CSV file', file_types=(("CSV Files", "*.csv"),))
            window['data_set'].update(value=selected_file_path)

    window.close()


if __name__ == '__main__':
    main()
