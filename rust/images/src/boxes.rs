//! The ISOBMFF boxes an AVIF still image uses (ISO 14496-12, 23008-12 and
//! the AV1 ISOBMFF binding): `ftyp`, then `meta` with `hdlr`, `pitm`,
//! `iloc`, `iinf`, `iref`, `iprp` (`ipco` holding `av1C`, `ispe`, `auxC`,
//! `colr`, `pixi`; `ipma` binding them to items), and the item bytes in
//! `mdat` or `idat`. Every size is checked against what is left of the
//! buffer before it is used, so a hostile length is an error, never a
//! panic or an allocation.

use crate::{Error, Result};

/// What the container says about the image, before any AV1 decoding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Properties {
    /// `ispe`: the image size the container declares.
    pub width: u32,
    pub height: u32,
    /// `av1C`: profile, level and the sequence flags of the primary item.
    pub av1c: [u8; 4],
    /// `pixi`: bits per channel, when declared.
    pub bits: Option<u8>,
    /// `colr` of type `nclx`: primaries, transfer, matrix and full range.
    pub nclx: Option<(u16, u16, u16, bool)>,
}

/// The parsed container: the AV1 bytes of the primary item and of its
/// alpha item, borrowed from the file.
#[derive(Debug)]
pub struct Container<'a> {
    pub primary: &'a [u8],
    pub alpha: Option<&'a [u8]>,
    pub premultiplied_alpha: bool,
    pub properties: Properties,
}

/// One box header: its type, the payload and where it started.
struct Box<'a> {
    name: [u8; 4],
    body: &'a [u8],
    at: usize,
}

/// A reader over the bytes, every step bounds-checked.
struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
    /// offset of `buf[0]` in the whole file, for error messages.
    base: usize,
}

impl<'a> Cursor<'a> {
    fn new(buf: &'a [u8], base: usize) -> Self {
        Self { buf, pos: 0, base }
    }

    fn left(&self) -> usize {
        self.buf.len() - self.pos
    }

    fn take(&mut self, n: usize, what: &'static str) -> Result<&'a [u8]> {
        if n > self.left() {
            return Err(Error::Truncated {
                at: self.base + self.pos,
                what,
            });
        }
        let s = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }

    fn u8(&mut self, what: &'static str) -> Result<u8> {
        Ok(self.take(1, what)?[0])
    }

    fn u16(&mut self, what: &'static str) -> Result<u16> {
        let b = self.take(2, what)?;
        Ok(u16::from_be_bytes([b[0], b[1]]))
    }

    fn u32(&mut self, what: &'static str) -> Result<u32> {
        let b = self.take(4, what)?;
        Ok(u32::from_be_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn u64(&mut self, what: &'static str) -> Result<u64> {
        let b = self.take(8, what)?;
        Ok(u64::from_be_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    /// An integer of `size` bytes (0, 4 or 8), as `iloc` encodes them.
    fn sized(&mut self, size: u8, what: &'static str) -> Result<u64> {
        match size {
            0 => Ok(0),
            4 => Ok(u64::from(self.u32(what)?)),
            8 => self.u64(what),
            _ => Err(Error::Malformed("iloc field size is not 0, 4 or 8")),
        }
    }

    /// The next box, or `None` at the end; a box larger than what is left
    /// is a hostile size.
    fn next_box(&mut self) -> Result<Option<Box<'a>>> {
        if self.left() == 0 {
            return Ok(None);
        }
        let at = self.base + self.pos;
        let size32 = self.u32("box size")?;
        let name_bytes = self.take(4, "box type")?;
        let name = [name_bytes[0], name_bytes[1], name_bytes[2], name_bytes[3]];
        let (size, header) = match size32 {
            0 => (self.left() as u64 + 8, 8usize),
            1 => (self.u64("box largesize")?, 16usize),
            n => (u64::from(n), 8usize),
        };
        let available = self.left() + header;
        if size < header as u64 || size > available as u64 {
            return Err(Error::HostileSize {
                name: String::from_utf8_lossy(&name).into_owned(),
                declared: size,
                available,
            });
        }
        let body = self.take(size as usize - header, "box body")?;
        Ok(Some(Box { name, body, at }))
    }

    /// A full box's version and flags.
    fn full(&mut self) -> Result<(u8, u32)> {
        let v = self.u8("fullbox version")?;
        let f = self.take(3, "fullbox flags")?;
        Ok((v, u32::from_be_bytes([0, f[0], f[1], f[2]])))
    }
}

/// Where an item's bytes are: in the file (`mdat`) or in `idat`.
#[derive(Clone, Copy, Debug)]
struct Extent {
    in_idat: bool,
    offset: u64,
    length: u64,
}

#[derive(Default)]
struct Meta<'a> {
    primary_id: Option<u32>,
    /// item id -> its extents, in order.
    locations: Vec<(u32, Vec<Extent>)>,
    /// item id -> item type.
    items: Vec<(u32, [u8; 4])>,
    /// `auxl` references: (from item, to item).
    auxl: Vec<(u32, u32)>,
    /// `prem` references: (alpha item, color item it premultiplies).
    prem: Vec<(u32, u32)>,
    /// `ipco` entries in order, 1-based when `ipma` names them.
    ipco: Vec<Box<'a>>,
    /// item id -> property indices.
    ipma: Vec<(u32, Vec<u16>)>,
    idat: Option<&'a [u8]>,
}

/// Parses the container without decoding anything.
pub fn parse(bytes: &[u8]) -> Result<Container<'_>> {
    let mut top = Cursor::new(bytes, 0);
    let first = top.next_box()?.ok_or(Error::NotAvif("empty"))?;
    if &first.name != b"ftyp" {
        return Err(Error::NotAvif("no ftyp box first"));
    }
    if first.body.len() < 8 {
        return Err(Error::Truncated {
            at: first.at,
            what: "ftyp brands",
        });
    }
    let avif_brand = first.body[8..]
        .chunks(4)
        .chain(std::iter::once(&first.body[..4]))
        .any(|b| b == b"avif" || b == b"avis" || b == b"mif1");
    if !avif_brand {
        return Err(Error::NotAvif("no avif, avis or mif1 brand"));
    }
    let mut meta = None;
    while let Some(b) = top.next_box()? {
        if &b.name == b"meta" {
            meta = Some(parse_meta(b)?);
            break;
        }
    }
    let meta = meta.ok_or(Error::Missing("meta"))?;
    let primary_id = meta.primary_id.ok_or(Error::Missing("pitm"))?;
    let kind = |id: u32| meta.items.iter().find(|(i, _)| *i == id).map(|(_, t)| *t);
    if kind(primary_id) != Some(*b"av01") {
        return Err(Error::Malformed("primary item is not an av01 item"));
    }
    let primary = item_bytes(bytes, &meta, primary_id)?;
    let properties = properties(&meta, primary_id)?;
    let alpha_id = meta
        .auxl
        .iter()
        .filter(|(_, to)| *to == primary_id)
        .map(|(from, _)| *from)
        .find(|&id| kind(id) == Some(*b"av01") && is_alpha(&meta, id));
    let alpha = match alpha_id {
        Some(id) => Some(item_bytes(bytes, &meta, id)?),
        None => None,
    };
    let premultiplied_alpha = alpha_id.is_some_and(|a| {
        meta.prem
            .iter()
            .any(|(from, to)| *from == a && *to == primary_id)
    });
    Ok(Container {
        primary,
        alpha,
        premultiplied_alpha,
        properties,
    })
}

fn parse_meta(b: Box<'_>) -> Result<Meta<'_>> {
    let mut c = Cursor::new(b.body, b.at + 8);
    c.full()?;
    let mut meta = Meta::default();
    while let Some(inner) = c.next_box()? {
        match &inner.name {
            b"pitm" => {
                let mut c = Cursor::new(inner.body, inner.at + 8);
                let (v, _) = c.full()?;
                meta.primary_id = Some(if v == 0 {
                    u32::from(c.u16("pitm item id")?)
                } else {
                    c.u32("pitm item id")?
                });
            }
            b"iloc" => meta.locations = parse_iloc(inner)?,
            b"iinf" => meta.items = parse_iinf(inner)?,
            b"iref" => parse_iref(inner, &mut meta)?,
            b"iprp" => parse_iprp(inner, &mut meta)?,
            b"idat" => meta.idat = Some(inner.body),
            _ => {}
        }
    }
    Ok(meta)
}

fn parse_iloc(b: Box<'_>) -> Result<Vec<(u32, Vec<Extent>)>> {
    let mut c = Cursor::new(b.body, b.at + 8);
    let (version, _) = c.full()?;
    let sizes = c.u16("iloc sizes")?;
    let offset_size = (sizes >> 12) as u8;
    let length_size = ((sizes >> 8) & 0xf) as u8;
    let base_offset_size = ((sizes >> 4) & 0xf) as u8;
    let index_size = if version >= 1 { (sizes & 0xf) as u8 } else { 0 };
    let count = if version < 2 {
        u32::from(c.u16("iloc item count")?)
    } else {
        c.u32("iloc item count")?
    };
    let mut out = Vec::new();
    for _ in 0..count {
        let id = if version < 2 {
            u32::from(c.u16("iloc item id")?)
        } else {
            c.u32("iloc item id")?
        };
        let construction = if version >= 1 {
            c.u16("iloc construction method")? & 0xf
        } else {
            0
        };
        c.u16("iloc data reference index")?;
        let base = c.sized(base_offset_size, "iloc base offset")?;
        let n = c.u16("iloc extent count")?;
        let per_extent =
            usize::from(index_size) + usize::from(offset_size) + usize::from(length_size);
        // an extent with no offset and no length can never name bytes
        // (item_bytes refuses len == 0), and a run of them is a hostile count
        // the payload would not bound.
        if per_extent == 0 && n > 1 {
            return Err(Error::Malformed(
                "iloc extents carry neither offset nor length",
            ));
        }
        if usize::from(n) * per_extent > c.left() {
            return Err(Error::Truncated {
                at: c.base + c.pos,
                what: "iloc extents",
            });
        }
        let mut extents = Vec::with_capacity(usize::from(n));
        for _ in 0..n {
            if index_size > 0 {
                c.sized(index_size, "iloc extent index")?;
            }
            let offset = c.sized(offset_size, "iloc extent offset")?;
            let length = c.sized(length_size, "iloc extent length")?;
            extents.push(Extent {
                in_idat: construction == 1,
                offset: base.saturating_add(offset),
                length,
            });
        }
        if construction > 1 {
            return Err(Error::Unsupported("iloc construction method 2"));
        }
        out.push((id, extents));
    }
    Ok(out)
}

fn parse_iinf(b: Box<'_>) -> Result<Vec<(u32, [u8; 4])>> {
    let mut c = Cursor::new(b.body, b.at + 8);
    let (version, _) = c.full()?;
    let count = if version == 0 {
        u32::from(c.u16("iinf entry count")?)
    } else {
        c.u32("iinf entry count")?
    };
    let mut out = Vec::new();
    for _ in 0..count {
        let infe = c.next_box()?.ok_or(Error::Truncated {
            at: c.base + c.pos,
            what: "infe",
        })?;
        if &infe.name != b"infe" {
            continue;
        }
        let mut ic = Cursor::new(infe.body, infe.at + 8);
        let (v, _) = ic.full()?;
        if v < 2 {
            return Err(Error::Unsupported("infe version below 2"));
        }
        let id = if v == 2 {
            u32::from(ic.u16("infe item id")?)
        } else {
            ic.u32("infe item id")?
        };
        ic.u16("infe protection index")?;
        let t = ic.take(4, "infe item type")?;
        out.push((id, [t[0], t[1], t[2], t[3]]));
    }
    Ok(out)
}

fn parse_iref(b: Box<'_>, meta: &mut Meta<'_>) -> Result<()> {
    let mut c = Cursor::new(b.body, b.at + 8);
    let (version, _) = c.full()?;
    while let Some(r) = c.next_box()? {
        let mut rc = Cursor::new(r.body, r.at + 8);
        let read = |rc: &mut Cursor<'_>, what| -> Result<u32> {
            Ok(if version == 0 {
                u32::from(rc.u16(what)?)
            } else {
                rc.u32(what)?
            })
        };
        let from = read(&mut rc, "iref from item")?;
        let n = rc.u16("iref reference count")?;
        for _ in 0..n {
            let to = read(&mut rc, "iref to item")?;
            match &r.name {
                b"auxl" => meta.auxl.push((from, to)),
                b"prem" => meta.prem.push((from, to)),
                _ => {}
            }
        }
    }
    Ok(())
}

fn parse_iprp<'a>(b: Box<'a>, meta: &mut Meta<'a>) -> Result<()> {
    let mut c = Cursor::new(b.body, b.at + 8);
    while let Some(inner) = c.next_box()? {
        match &inner.name {
            b"ipco" => {
                let mut pc = Cursor::new(inner.body, inner.at + 8);
                while let Some(p) = pc.next_box()? {
                    meta.ipco.push(p);
                }
            }
            b"ipma" => {
                let mut ac = Cursor::new(inner.body, inner.at + 8);
                let (version, flags) = ac.full()?;
                let count = ac.u32("ipma entry count")?;
                for _ in 0..count {
                    let id = if version == 0 {
                        u32::from(ac.u16("ipma item id")?)
                    } else {
                        ac.u32("ipma item id")?
                    };
                    let n = ac.u8("ipma association count")?;
                    let mut props = Vec::with_capacity(usize::from(n));
                    for _ in 0..n {
                        let index = if flags & 1 == 1 {
                            ac.u16("ipma property index")? & 0x7fff
                        } else {
                            u16::from(ac.u8("ipma property index")? & 0x7f)
                        };
                        props.push(index);
                    }
                    meta.ipma.push((id, props));
                }
            }
            _ => {}
        }
    }
    Ok(())
}

/// The property boxes bound to `id`, in `ipma` order.
fn props_of<'m, 'a>(meta: &'m Meta<'a>, id: u32) -> impl Iterator<Item = &'m Box<'a>> {
    meta.ipma
        .iter()
        .filter(move |(i, _)| *i == id)
        .flat_map(|(_, p)| p.iter())
        .filter_map(|&index| {
            index
                .checked_sub(1)
                .and_then(|i| meta.ipco.get(usize::from(i)))
        })
}

fn is_alpha(meta: &Meta<'_>, id: u32) -> bool {
    props_of(meta, id).any(|p| {
        &p.name == b"auxC"
            && p.body.len() > 4
            && p.body[4..].starts_with(b"urn:mpeg:mpegB:cicp:systems:auxiliary:alpha")
    })
}

fn properties(meta: &Meta<'_>, id: u32) -> Result<Properties> {
    let mut av1c = None;
    let mut size = None;
    let mut bits = None;
    let mut nclx = None;
    for p in props_of(meta, id) {
        let mut c = Cursor::new(p.body, p.at + 8);
        match &p.name {
            b"av1C" => {
                let b = c.take(4, "av1C")?;
                if b[0] & 0x80 == 0 {
                    return Err(Error::Malformed("av1C marker bit is not set"));
                }
                av1c = Some([b[0], b[1], b[2], b[3]]);
            }
            b"ispe" => {
                c.full()?;
                size = Some((c.u32("ispe width")?, c.u32("ispe height")?));
            }
            b"pixi" => {
                c.full()?;
                let n = c.u8("pixi channel count")?;
                if n > 0 {
                    bits = Some(c.u8("pixi bits")?);
                }
            }
            b"colr" => {
                let kind = c.take(4, "colr type")?;
                if kind != b"nclx" {
                    continue;
                }
                let pri = c.u16("colr primaries")?;
                let trc = c.u16("colr transfer")?;
                let mtx = c.u16("colr matrix")?;
                let full = c.u8("colr range")? & 0x80 != 0;
                nclx = Some((pri, trc, mtx, full));
            }
            _ => {}
        }
    }
    let av1c = av1c.ok_or(Error::Missing("av1C"))?;
    let (width, height) = size.ok_or(Error::Missing("ispe"))?;
    if width == 0 || height == 0 {
        return Err(Error::Malformed("ispe declares an empty image"));
    }
    Ok(Properties {
        width,
        height,
        av1c,
        bits,
        nclx,
    })
}

/// The bytes of item `id`, concatenated from its extents; a single extent
/// is borrowed, which is every file urna writes.
fn item_bytes<'a>(file: &'a [u8], meta: &Meta<'a>, id: u32) -> Result<&'a [u8]> {
    let (_, extents) = meta
        .locations
        .iter()
        .find(|(i, _)| *i == id)
        .ok_or(Error::Missing("iloc entry of the item"))?;
    let [extent] = extents.as_slice() else {
        return Err(Error::Unsupported("an item spread over several extents"));
    };
    let source = if extent.in_idat {
        meta.idat.ok_or(Error::Missing("idat"))?
    } else {
        file
    };
    let start = usize::try_from(extent.offset).map_err(|_| Error::Malformed("iloc offset"))?;
    let len = usize::try_from(extent.length).map_err(|_| Error::Malformed("iloc length"))?;
    let end = start
        .checked_add(len)
        .filter(|&e| e <= source.len())
        .ok_or(Error::HostileSize {
            name: "iloc".into(),
            declared: extent.length,
            available: source.len().saturating_sub(start),
        })?;
    if len == 0 {
        return Err(Error::Malformed("empty item"));
    }
    Ok(&source[start..end])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The smallest container the parser accepts: ftyp, meta with pitm,
    /// iloc into mdat, iinf, iprp with av1C and ispe, and the payload.
    fn minimal(payload: &[u8]) -> Vec<u8> {
        fn boxed(name: &[u8; 4], body: &[u8]) -> Vec<u8> {
            let mut v = ((body.len() + 8) as u32).to_be_bytes().to_vec();
            v.extend_from_slice(name);
            v.extend_from_slice(body);
            v
        }
        fn build(payload: &[u8], offset: u32) -> Vec<u8> {
            let ftyp = boxed(b"ftyp", b"avif\0\0\0\0avifmif1");
            let pitm = boxed(b"pitm", &[0, 0, 0, 0, 0, 1]);
            let infe = boxed(b"infe", &[2, 0, 0, 0, 0, 1, 0, 0, b'a', b'v', b'0', b'1']);
            let mut iinf_body = vec![0, 0, 0, 0, 0, 1];
            iinf_body.extend(infe);
            let iinf = boxed(b"iinf", &iinf_body);
            let av1c = boxed(b"av1C", &[0x81, 0x04, 0x0c, 0x00]);
            let mut ispe_body = vec![0, 0, 0, 0];
            ispe_body.extend(8u32.to_be_bytes());
            ispe_body.extend(4u32.to_be_bytes());
            let ispe = boxed(b"ispe", &ispe_body);
            let mut ipco_body = av1c;
            ipco_body.extend(ispe);
            let ipco = boxed(b"ipco", &ipco_body);
            let ipma = boxed(b"ipma", &[0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 2, 0x81, 0x02]);
            let mut iprp_body = ipco;
            iprp_body.extend(ipma);
            let iprp = boxed(b"iprp", &iprp_body);
            // iloc version 0: offset and length 4 bytes, no base; one item, one extent
            let mut iloc_body = vec![0, 0, 0, 0, 0x44, 0x00, 0, 1, 0, 1, 0, 0, 0, 1];
            iloc_body.extend(offset.to_be_bytes());
            iloc_body.extend((payload.len() as u32).to_be_bytes());
            let iloc = boxed(b"iloc", &iloc_body);
            let mut meta_body = vec![0, 0, 0, 0];
            meta_body.extend(pitm);
            meta_body.extend(iloc);
            meta_body.extend(iinf);
            meta_body.extend(iprp);
            let meta = boxed(b"meta", &meta_body);
            let mdat = boxed(b"mdat", payload);
            let mut file = ftyp;
            file.extend(meta);
            file.extend(mdat);
            file
        }
        // the payload sits after every box header; its offset is read off a
        // first build, whose lengths the offset value does not change.
        let probe = build(payload, 0);
        let offset = (probe.len() - payload.len()) as u32;
        build(payload, offset)
    }

    #[test]
    fn reads_the_primary_item_and_its_properties() {
        let file = minimal(b"OBU!");
        let c = parse(&file).expect("parses");
        assert_eq!(c.primary, b"OBU!");
        assert!(c.alpha.is_none());
        assert_eq!((c.properties.width, c.properties.height), (8, 4));
        assert_eq!(c.properties.av1c, [0x81, 0x04, 0x0c, 0x00]);
    }

    #[test]
    fn truncation_hostile_sizes_and_missing_boxes_are_errors_not_panics() {
        let file = minimal(b"OBU!");
        for cut in 0..file.len() {
            let r = parse(&file[..cut]);
            assert!(r.is_err(), "cut at {cut} parsed");
        }
        let mut hostile = file.clone();
        let pitm = hostile.windows(4).position(|w| w == b"pitm").expect("pitm");
        hostile[pitm - 4..pitm].copy_from_slice(&u32::MAX.to_be_bytes());
        assert!(matches!(parse(&hostile), Err(Error::HostileSize { .. })));
        let mut no_av1c = file.clone();
        let i = no_av1c.windows(4).position(|w| w == b"av1C").expect("av1C");
        no_av1c[i..i + 4].copy_from_slice(b"zzzz");
        assert!(matches!(parse(&no_av1c), Err(Error::Missing("av1C"))));
        assert!(matches!(
            parse(b"not an avif"),
            Err(Error::HostileSize { .. })
        ));
        assert!(matches!(parse(b"\0\0"), Err(Error::Truncated { .. })));
        // iloc v0, no field sizes, one item declaring 65535 extents: refused
        // before any allocation; with 4-byte fields the count must fit.
        let empty = [0, 0, 0, 0, 0x00, 0x00, 0, 1, 0, 1, 0, 0, 0xff, 0xff];
        let iloc = |body| Box {
            name: *b"iloc",
            body,
            at: 0,
        };
        assert!(matches!(parse_iloc(iloc(&empty)), Err(Error::Malformed(_))));
        let mut short = empty;
        short[4] = 0x44;
        assert!(matches!(
            parse_iloc(iloc(&short)),
            Err(Error::Truncated { .. })
        ));
        assert!(matches!(
            parse(&[0, 0, 0, 8, b'f', b'r', b'e', b'e']),
            Err(Error::NotAvif(_))
        ));
    }
}
