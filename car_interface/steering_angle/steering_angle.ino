// Global variables
const byte encoderA = 2;
const byte encoderB = 3;

// assinge ram for variable
volatile int encoderPos = 0;

// serial read val
char pyRequest;

void setup() {
	pinMode(encoderA, INPUT_PULLUP);
	pinMode(encoderB, INPUT_PULLUP);
	attachInterrupt(digitalPinToInterrupt(encoderA), encoderChange, CHANGE);
	
	Serial.begin(9600);
}

void loop() {
	// only send data when requested
	if (Serial.available() > 0 ) {
		pyRequest = Serial.read();
		switch (pyRequest) {
			case '1':
				Serial.println(encoderPos, DEC);
				break;
			case '2':
				encoderPos = 0;
				break;
			case '3':
				// divide by 2
				encoderPos = encoderPos >> 1;
				break;
		}
	}
}

void encoderChange() {
	if (digitalRead(encoderA) == digitalRead(encoderB)) {
		encoderPos++;
	} else {
		encoderPos--;
	}
}
	

