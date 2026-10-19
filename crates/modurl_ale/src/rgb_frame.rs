use burn_std::{
    AccessError, AccessPolicy, AllocationController, AllocationProperty, Bytes, Reader, Writer,
};
use std::{
    mem::MaybeUninit,
    sync::{Arc, OnceLock},
};

struct RgbFrame {
    pixels: Vec<u8>,
    palette: Arc<[u32; 256]>,
    rgb: OnceLock<Bytes>,
}

impl RgbFrame {
    fn materialize(&self, policy: AccessPolicy) -> Result<&Bytes, AccessError> {
        if self.rgb.get().is_none() && !policy.copy_allowed() {
            return Err(AccessError::WouldCopy);
        }
        Ok(self.rgb.get_or_init(|| {
            let rgb = self
                .pixels
                .iter()
                .map(|&index| {
                    let color = self.palette[usize::from(index)];
                    [(color >> 16) as u8, (color >> 8) as u8, color as u8]
                })
                .collect::<Vec<_>>();
            Bytes::from_elems(rgb)
        }))
    }
}

impl AllocationController for RgbFrame {
    fn alloc_align(&self) -> usize {
        std::mem::align_of::<u8>()
    }

    fn property(&self) -> AllocationProperty {
        AllocationProperty::Other
    }

    fn capacity(&self) -> usize {
        self.pixels.len() * 3
    }

    fn memory(&self, policy: AccessPolicy) -> Result<&[MaybeUninit<u8>], AccessError> {
        let bytes = self.materialize(policy)?.read(Reader::new().no_copy())?;
        // SAFETY: The owned RGB buffer is fully initialized and stays live while self is borrowed.
        // u8 and MaybeUninit<u8> have the same size and alignment.
        Ok(unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast(), bytes.len()) })
    }

    unsafe fn memory_mut(
        &mut self,
        policy: AccessPolicy,
    ) -> Result<&mut [MaybeUninit<u8>], AccessError> {
        self.materialize(policy)?;
        let Some(rgb) = self.rgb.get_mut() else {
            unreachable!("RGB data was initialized before mutable access");
        };
        let bytes = rgb.write(Writer::new().no_copy())?;
        // SAFETY: The buffer is fully initialized, its length does not change, and &mut self
        // provides exclusive access. u8 and MaybeUninit<u8> have the same size and alignment.
        Ok(unsafe { std::slice::from_raw_parts_mut(bytes.as_mut_ptr().cast(), bytes.len()) })
    }
}

/// Captures indexed pixels `[pixel_count]` and a palette `[256]` as interleaved RGB bytes `[pixel_count * 3]`.
/// Owns the pixels and retains an immutable palette. Conversion occurs on the first read or write.
pub(crate) fn snapshot(pixels: Vec<u8>, palette: Arc<[u32; 256]>) -> Bytes {
    let len = pixels.len() * 3;
    let frame = RgbFrame {
        pixels,
        palette,
        rgb: OnceLock::new(),
    };
    // SAFETY: Every successful access exposes exactly len initialized RGB bytes. The controller
    // owns the indexed pixels, retains the palette, and keeps materialized storage live until drop.
    unsafe { Bytes::from_controller(Box::new(frame), len) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metadata_does_not_convert_pixels_and_reads_match_rgb_order() {
        let mut palette = [0; 256];
        palette[1] = 0x00123456;
        palette[255] = 0x00abcdef;
        let bytes = snapshot(vec![1, 255, 0], Arc::new(palette));
        assert_eq!(bytes.len(), 9);
        assert_eq!(bytes.capacity(), 9);
        assert_eq!(bytes.align(), 1);
        assert_eq!(bytes.property(), AllocationProperty::Other);
        assert!(matches!(
            bytes.read(Reader::new().no_copy()),
            Err(AccessError::WouldCopy)
        ));
        assert_eq!(
            bytes.read(Reader::new()).unwrap(),
            [0x12, 0x34, 0x56, 0xab, 0xcd, 0xef, 0, 0, 0]
        );
        assert_eq!(bytes.read(Reader::new().no_copy()).unwrap().len(), 9);
    }

    #[test]
    fn snapshots_keep_their_palette_and_mutations_do_not_change_clones() {
        let mut palette = Arc::new([0x00123456; 256]);
        let mut bytes = snapshot(vec![0, 1], palette.clone());
        Arc::make_mut(&mut palette)[0] = 0x00ffffff;
        assert!(matches!(
            bytes.write(Writer::new().no_copy()),
            Err(AccessError::WouldCopy)
        ));
        assert_eq!(
            bytes.read(Reader::new()).unwrap(),
            [0x12, 0x34, 0x56, 0x12, 0x34, 0x56]
        );
        let copy = bytes.clone();
        bytes.write(Writer::new().no_copy()).unwrap()[0] = 99;
        assert_eq!(copy.read(Reader::new()).unwrap()[0], 0x12);
        assert_eq!(bytes.read(Reader::new()).unwrap()[0], 99);
    }

    #[test]
    fn concurrent_reads_initialize_one_stable_snapshot() {
        let bytes = Arc::new(snapshot(vec![7; 128], Arc::new([0x00abcdef; 256])));
        std::thread::scope(|scope| {
            for _ in 0..4 {
                let bytes = &bytes;
                scope.spawn(move || {
                    assert_eq!(
                        bytes.read(Reader::new()).unwrap(),
                        [0xab, 0xcd, 0xef].repeat(128)
                    );
                });
            }
        });
        assert_eq!(bytes.read(Reader::new().no_copy()).unwrap().len(), 384);
    }

    #[test]
    fn tensor_construction_and_metadata_preserve_deferred_conversion() {
        use burn::tensor::{DType, Device, Int, Tensor, TensorData};
        let bytes = snapshot(vec![0, 7], Arc::new([0x00123456; 256]));
        let tensor = Tensor::<3, Int>::from_data(
            TensorData::from_bytes(bytes, [1, 2, 3], DType::U8),
            (&Device::flex(), DType::U8),
        );
        assert_eq!(tensor.dims(), [1, 2, 3]);
        assert_eq!(tensor.dtype(), DType::U8);
        let data = tensor.into_data();
        assert!(matches!(
            data.bytes().read(Reader::new().no_copy()),
            Err(AccessError::WouldCopy)
        ));
        assert_eq!(
            data.try_to_vec::<u8>().unwrap(),
            [0x12, 0x34, 0x56, 0x12, 0x34, 0x56]
        );
    }

    #[test]
    fn empty_snapshot_has_no_rgb_bytes() {
        let bytes = snapshot(Vec::new(), Arc::new([0; 256]));
        assert_eq!(bytes.len(), 0);
        assert!(bytes.read(Reader::new()).unwrap().is_empty());
    }
}
