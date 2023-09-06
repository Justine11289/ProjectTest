#include <Wire.h>
#include <MPU6050.h>

MPU6050 mpu1;//0x68
MPU6050 mpu2;//0x69

// Timers
float timeStep = 0.01;

// Pitch, Roll and Yaw values
float pitch1 = 0,roll1 = 0,yaw1 = 0;
float pitch2 = 0,roll2 = 0,yaw2 = 0;

bool shouldTransmit = false;
void setup() 
{
  delay(500);
  Serial.begin(115200);

  // Initialize MPU6050 1
  while(!mpu1.begin(MPU6050_SCALE_250DPS, MPU6050_RANGE_2G,0x68))
  {
    Serial.println("Could not find a valid MPU6050 sensor 1(0x68), check wiring!");
    delay(500);
  }

  // Calibrate gyroscope. The calibration must be at rest.
  mpu1.calibrateGyro();
  
  // Set threshold sensivty. 
  mpu1.setThreshold(0);
  
  delay(500);
  
   // Initialize MPU6050 2
  while(!mpu2.begin(MPU6050_SCALE_250DPS, MPU6050_RANGE_2G,0x69))
  {
    Serial.println("Could not find a valid MPU6050 sensor 2(0x69), check wiring!");
    delay(500);
  }
  
  // Calibrate gyroscope. The calibration must be at rest.
  mpu2.calibrateGyro();
  
  // Set threshold sensivty.
  mpu2.setThreshold(0);
  
  Serial.println("start computing!");
}

void loop(){
  if (Serial.available() > 0) {
    char receivedChar = Serial.read();
    if (receivedChar == 's') {
        shouldTransmit = true;
        Serial.println("Start!");
    } else if (receivedChar == 'e') {
        shouldTransmit = false;
        Serial.println("END!");
    }
  }
  
  if (shouldTransmit) {
    // Read normalized values
    Vector norm1 = mpu1.readNormalizeGyro();
    Vector norm2 = mpu2.readNormalizeGyro();
  
    // Calculate Pitch, Roll and Yaw
    pitch1 = norm1.YAxis * timeStep;//degree
    roll1 = norm1.XAxis * timeStep;
    yaw1 = norm1.ZAxis * timeStep;
    
    pitch2 = norm2.YAxis * timeStep;//degree
    roll2 = norm2.XAxis * timeStep;
    yaw2 = norm2.ZAxis * timeStep;
    
    Serial.print("(");
    Serial.print(pitch1);
    Serial.print(",");
    Serial.print(roll1);  
    Serial.print(",");
    Serial.print(yaw1);;
    Serial.print(")");
    Serial.print("(");
    Serial.print(pitch2);
    Serial.print(",");
    Serial.print(roll2);  
    Serial.print(",");
    Serial.println(yaw2);
    Serial.print(")");
  
    delay(20);
  }
}
  
