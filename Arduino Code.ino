#include <Servo.h>
#include <SoftwareSerial.h>
#include <Stepper.h>

const int stepsPerRevolution = 2048;

Stepper MyStepper(stepsPerRevolution, 8, 10, 9, 11);

SoftwareSerial mySerial(2, 3); // Raspberry Pi Communication

// Servo Motor Setup
#define SERVO_PIN 6
Servo wasteGate;
const int openAngle = 90;  // Open position
const int closeAngle = 180; // Close position

// Stepper Control Variables
const int stepsPerBin = 575;
int currentBin = 0;
bool isProcessing = false; // Flag to ensure one command at a time
int count = 0;
bool wasteGateOperated = false;  // Flag to prevent repeated wastegate operation

// Move to specific bin
void moveToBin(int binIndex) {
    if (binIndex < 0 || binIndex > 2) return;

    int stepsToMove = (binIndex - currentBin)*stepsPerBin;
    MyStepper.step(stepsToMove);
    Serial.print(stepsToMove);
    delay(3000); // Give time for stepper to move
    currentBin = binIndex;
}

// Open and close waste gate
void openWasteGate() {
    wasteGate.write(openAngle);  
    delay(500);
    wasteGate.write(closeAngle);
    delay(500);
    wasteGateOperated = true;  // Mark that the wastegate has been operated
}

void setup() {
    Serial.begin(9600);
    MyStepper.setSpeed(6);
    mySerial.begin(9600); // Raspberry Pi Communication
    wasteGate.attach(SERVO_PIN);
    MyStepper.step(6);  // Initial stepper movement

    // Initialize stepper motor to bin 0
    moveToBin(0);
    wasteGate.write(closeAngle); 
}

void loop() {
    if (!isProcessing && mySerial.available()) {
        int binIndex = mySerial.parseInt();
        // Clear any leftover data in the serial buffer
        while (mySerial.available()) {
            mySerial.read(); // Clear any extra characters from the buffer
        }

        if (binIndex >= 0 && binIndex < 3) {
            isProcessing = true;  // Mark as busy
            wasteGateOperated = false; // Reset the wastegate operation flag

            Serial.print("Processing Bin: ");
            Serial.println(binIndex);

            // Move to the desired bin
            moveToBin(binIndex);

            // Operate the wastegate (if not already operated)
            if (!wasteGateOperated) {
                openWasteGate();
            }

            // Once the wastegate is operated, return stepper to bin 0
            moveToBin(0);

            isProcessing = false;  // Mark as ready for next data
        }
    }
}
