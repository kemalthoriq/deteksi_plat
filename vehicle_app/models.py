from django.db import models

class VehicleImage(models.Model):
    image = models.ImageField(upload_to='license_plates/')
    processed_image = models.ImageField(upload_to='processed_plates/', null=True, blank=True)  # Tambahkan field ini
    detected_plate = models.CharField(max_length=50, blank=True, null=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.detected_plate or "Plate not detected"
